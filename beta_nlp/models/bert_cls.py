import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from munch import munchify
from sklearn import metrics
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm, trange
from transformers import (
    AdamW,
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from beta_nlp.utils.common import (
    ensureDir,
    print_dict_as_table,
    print_transformer,
    timeit,
)
from beta_nlp.utils.optimization import warmup_linear
from beta_nlp.utils.textloader import convert_df_to_dataset


class BertModel(object):
    def __init__(self, config):
        self.config = config
        self.init_config()
        self.init_random_seeds()
        self.init_bert()

    def init_config(self):
        self.args = munchify(self.config)
        self.pretrained_model = self.args.pretrained_model
        self.device = self.args.device
        self.n_gpu = (
            len(self.args.gpu_ids.split(","))
            if "gpu_ids" in self.config
            else torch.cuda.device_count()
        )
        if "gpu_ids" in self.config:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config["gpu_ids"]

    def init_bert(self):
        self.model = BertForSequenceClassification.from_pretrained(
            self.pretrained_model, num_labels=self.args.num_labels,
        )
        print_transformer(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer)
        if self.args.fp16:
            self.model.half()
        self.model.to(self.device)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

    def init_optimizer(self, n_examples):
        num_train_optimization_steps = (
            int(
                n_examples
                / self.args.batch_size
                / self.args.gradient_accumulation_steps
            )
            * self.args.epochs
        )
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer, FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use"
                    " distributed and fp16 training."
                )

            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                bias_correction=False,
                max_grad_norm=1.0,
            )
            if self.args.loss_scale == 0:
                self.optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                self.optimizer = FP16_Optimizer(
                    optimizer, static_loss_scale=self.args.loss_scale
                )
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                warmup=self.args.warmup_proportion,
                t_total=num_train_optimization_steps,
            )

        else:
            self.optimizer = AdamW(
                self.model.parameters(), lr=self.args.lr, correct_bias=False
            )
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_train_optimization_steps,
                num_training_steps=self.args.warmup_proportion
                * num_train_optimization_steps,
            )

    def init_random_seeds(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)

    def save_pretrained(self, path):
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(path)

    @timeit
    def train_an_epoch(self, train_dataloader):
        self.model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, input_mask, label_ids = batch
            if self.args.is_multilabel:
                logits = self.model(
                    input_ids, token_type_ids=None, attention_mask=input_mask,
                )[0]
                loss = F.binary_cross_entropy_with_logits(logits, label_ids.float())
            else:
                loss, logits = self.model(
                    input_ids,
                    token_type_ids=None,
                    attention_mask=input_mask,
                    labels=label_ids,
                )
                if self.n_gpu > 1:
                    loss = loss.mean()
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if self.args.fp16:
                    self.optimizer.backward(loss)
                else:
                    loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.tr_loss += loss.item()
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if self.args.fp16:
                    lr_this_step = self.args.learning_rate * warmup_linear(
                        self.iterations / self.num_train_optimization_steps,
                        self.args.warmup_proportion,
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr_this_step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.iterations += 1

    def train(self, train_set, dev_set):
        self.iterations, self.nb_tr_steps, self.tr_loss = 0, 0, 0
        self.best_valid_metric, self.unimproved_iters = 0, 0
        self.early_stop = False
        if self.args.gradient_accumulation_steps < 1:
            raise ValueError(
                "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                    self.args.gradient_accumulation_steps
                )
            )

        self.args.batch_size = (
            self.args.batch_size // self.args.gradient_accumulation_steps
        )
        self.init_optimizer(len(train_set))

        train_dataset = convert_df_to_dataset(
            train_set, self.tokenizer, self.args.max_seq_length
        )
        dev_dataset = convert_df_to_dataset(
            dev_set, self.tokenizer, self.args.max_seq_length
        )

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.args.batch_size,
        )
        dev_dataloader = DataLoader(
            dev_dataset,
            sampler=SequentialSampler(dev_dataset),
            batch_size=self.args.batch_size,
        )

        for epoch in trange(int(self.args.epochs), desc="Epoch"):
            self.train_an_epoch(train_dataloader)
            tqdm.write(
                f"[Epoch {epoch}] loss: {self.tr_loss}".format(
                    epoch, self.best_valid_metric
                )
            )
            self.tr_loss = 0
            eval_result = self.eval(dev_dataloader)
            # Update validation results
            if eval_result[self.args.valid_metric] > self.best_valid_metric:
                self.unimproved_iters = 0
                self.best_valid_metric = eval_result[self.args.valid_metric]
                print_dict_as_table(
                    eval_result,
                    tag=f"[Epoch {epoch}]performance on validation set",
                    columns=["metrics", "values"],
                )
                ensureDir(self.args.model_save_dir)
                self.save_pretrained(self.args.model_save_dir)
            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.args.patience:
                    self.early_stop = True
                    tqdm.write(
                        "Early Stopping. Epoch: {}, best_valid_metric ({}): {}".format(
                            epoch, self.args.valid_metric, self.best_valid_metric
                        )
                    )
                    break

    def test(self, test_set):
        """Get a evaluation result for a test set.

        Args:
            test_set:

        Returns:

        """
        test_dataset = convert_df_to_dataset(
            test_set, self.tokenizer, self.args.max_seq_length
        )

        test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=self.args.batch_size,
        )
        return self.eval(test_dataloader)

    def scores(self, test_dataloader):
        """Get predicted label scores for a test_dataloader

        Args:
            test_dataloader:

        Returns:
            ndarray: An array of predicted label scores.

        """
        self.model.eval()
        predicted_labels, target_labels = list(), list()
        for input_ids, input_mask, label_ids in tqdm(
            test_dataloader, desc="Evaluating"
        ):
            input_ids = input_ids.to(self.args.device)
            input_mask = input_mask.to(self.args.device)
            label_ids = label_ids.to(self.args.device)
            with torch.no_grad():
                logits = self.model(
                    input_ids, token_type_ids=None, attention_mask=input_mask
                )[0]
            if self.args.is_multilabel:
                predicted_labels.extend(
                    F.sigmoid(logits).round().long().cpu().detach().numpy()
                )
            else:
                predicted_labels.extend(
                    torch.argmax(logits, dim=1).cpu().detach().numpy()
                )
            target_labels.extend(label_ids.cpu().detach().numpy())
        return np.array(predicted_labels), np.array(target_labels)

    @timeit
    def eval(self, test_dataloader):
        """Get the evaluation performance of a test_dataloader

        Args:
            test_dataloader:

        Returns:
            dict: A result dict containing result of "accuracy"， "precision", "recall"
                and "F1".

        """
        # test loader tensor: input_ids, input_mask, segment_ids, label_ids
        predicted_labels, target_labels = self.scores(test_dataloader)

        if self.args.num_labels > 2:
            accuracy = metrics.accuracy_score(target_labels, predicted_labels)
            macro_precision = metrics.precision_score(
                target_labels, predicted_labels, average="macro"
            )
            macro_recall = metrics.recall_score(
                target_labels, predicted_labels, average="macro"
            )
            macro_f1 = metrics.f1_score(
                target_labels, predicted_labels, average="macro"
            )
            micro_precision = metrics.precision_score(
                target_labels, predicted_labels, average="micro"
            )
            micro_recall = metrics.recall_score(
                target_labels, predicted_labels, average="micro"
            )
            micro_f1 = metrics.f1_score(
                target_labels, predicted_labels, average="micro"
            )

            return {
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
            }

        else:
            accuracy = metrics.accuracy_score(target_labels, predicted_labels)
            precision = metrics.precision_score(
                target_labels, predicted_labels, average="binary"
            )
            recall = metrics.recall_score(
                target_labels, predicted_labels, average="binary"
            )
            f1 = metrics.f1_score(target_labels, predicted_labels, average="binary")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    @timeit
    def predict(self, test_set):
        """

        Args:
            test_set: list of :obj:InputExample

        Returns:
            ndarray: An array of predicted label scores.
        """
        test_dataset = convert_df_to_dataset(
            test_set, self.tokenizer, self.args.max_seq_length
        )

        test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=self.args.batch_size,
        )
        return self.scores(test_dataloader)[0]
