import os
import random

import numpy as np
from sklearn import metrics
from tqdm.auto import tqdm

import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from beta_nlp.utils.common import ensureDir, get_device, print_dict_as_table, timeit
from beta_nlp.utils.textloader import convert_df_to_ids
from munch import munchify
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

embedding_path_dict = {
    "googlenews": {
        "path": "../resources/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin",
        "format": "word2vec",
        "binary": True,
    },
    "glove": {
        "path": "../../resources/embeddings/glove.840B.300d/glove.840B.300d.txt",
        "format": "glove",
        "binary": "",
    },
    "glove_word2vec": {
        "path": "../../resources/glove.840B.300d.txt.word2vec",
        "format": "word2vec",
        "binary": False,
    },
    "wiki": {
        "path": "../../resources/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec",
        "format": "word2vec",
        "binary": False,
    },
    "paragram": {
        "path": "../../resources/embeddings/paragram_300_sl999/paragram_300_sl999.txt",
        "format": "",
        "binary": False,
    },
}


def mixup_data(x, y, device, alpha=1.0):

    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda"""
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def nonlinear_mixup_data(x, y=None, alpha=1.0):

    """Compute the nonlinear_mixup data. Return mixed inputs, pairs of targets, and lambda"""
    if alpha > 0.0:
        lam = np.random.beta(np.ones(x[0].shape) * alpha, np.ones(x[0].shape) * alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    lam = torch.tensor(lam, device=x.device, dtype=torch.float)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = None, None
    if y is not None:
        y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(
        pred, y_b
    )


def non_linear_mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(
        pred, y_b
    )


class TextCLS(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_length,
        embed_dim,
        num_class,
        label_dim,
        device,
        dropout,
        mixup="word",
    ):
        super().__init__()
        self.device = device
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.mixup = mixup
        self.num_class = num_class
        self.label_dim = label_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True).to(device)
        q, r = torch.qr(torch.randn(num_class, label_dim, requires_grad=False))
        pad = nn.ZeroPad2d(padding=(0, label_dim - num_class, 0, 0))
        self.label_embedding = pad(q).to(device)
        Ci = 1
        Co = 100
        Ks = [3, 4, 5]
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embed_dim)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.theta = nn.Linear(max_seq_length * embed_dim, label_dim).to(device)
        self.fc = nn.Linear(len(Ks) * Co, label_dim).to(device)
        self.cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def init_weights(self, embedding_list):
        initrange = 0.5
        if embedding_list:
            self.embedding.weight.data = torch.tensor(embedding_list).to(self.device)
        else:
            self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.theta.weight.data.uniform_(-initrange, initrange)
        self.theta.bias.data.zero_()

    def forward(self, input_ids, y=None):
        x = self.embedding(input_ids)  # (N, W, D)
        x, targets_a, targets_b, _ = nonlinear_mixup_data(x, y)
        if targets_a is not None and targets_b is not None:
            targets_a, targets_b = (
                self.label_embedding[targets_a],
                self.label_embedding[targets_b],
            )
            # print(x.dtype)
            theta = self.theta(
                x.view(-1, self.max_seq_length * self.embed_dim)
            )  # eq 10
            theta = torch.sigmoid(theta)
            # print(self.label_embedding.size())
            # print(targets_a.size())
            # print(theta.size())
            labels = theta * targets_a + (1 - theta) * targets_b  # eq 11

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [
            F.relu(conv(x)).squeeze(3) for conv in self.convs
        ]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        output = self.fc(x)  # (N, C)

        if y is not None:
            return output, labels
        else:
            return output

    def predict(self, input_ids):
        """Compute the label embeddings for input_ids, and calculate the most similar label_ids
        based on the cosine similarities between the predict label embeddings and the gold
        label embeddings.

        Args:
            input_ids:

        Returns:

        """
        with torch.no_grad():
            output = self.forward(input_ids)
        label_embedding = torch.stack([self.label_embedding] * output.size()[0]).view(
            -1, self.label_dim
        )
        output = torch.repeat_interleave(output, self.num_class, dim=0)
        scores = self.cosine(label_embedding, output)
        return torch.argmax(scores.view(-1, self.num_class), dim=1)


class NonLinearMixupModel(object):
    def __init__(self, train_set, config):
        self.build_vocab(train_set)
        self.config = config
        self.init_config()
        self.init_random_seeds()
        self.model = TextCLS(
            self.vocab_size + 1,
            self.args.max_seq_length,
            self.embed_dim,
            self.args.num_labels,
            self.args.label_dim,
            self.device,
            self.args.dropout,
        ).to(self.device)
        self.model.init_weights(self.embedding_list)

    def init_config(self):
        gpu_id, self.config["device"] = get_device(self.config)
        self.args = munchify(self.config)
        self.device = self.args.device
        self.n_gpu = (
            len(self.args.gpu_ids.split(","))
            if "gpu_ids" in self.config
            else torch.cuda.device_count()
        )
        if "gpu_ids" in self.config:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config["gpu_ids"]

    def init_weights(self):
        self.embedding.weight = torch.tensor(self.embedding_list).to(self.device)
        self.fc.bias.data.zero_()

    def init_optimizer(self,):
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.criterion = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

    def init_random_seeds(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)

    def save_pretrained(self, path):
        # model_to_save = (
        #     self.model.module if hasattr(self.model, "module") else self.model
        # )
        # model_to_save.save_pretrained(path)
        pass

    @timeit
    def train_an_epoch(self, train_dataloader):
        self.model.train()
        train_loss, train_acc = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.optimizer.zero_grad()
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, label_ids = batch
            if self.args.mixup:
                output, labels = self.model.forward(input_ids, label_ids)
                # print(self.criterion(output, labels).size())
                loss = -torch.mean(self.criterion(output, labels))
            else:
                output = self.model(input_ids)
                loss = -torch.mean(self.criterion(output, label_ids))
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            train_acc += (output.argmax(1) == label_ids).sum().item()
        self.scheduler.step()
        return train_loss / len(train_dataloader), train_acc / len(train_dataloader)

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
        self.init_optimizer()

        train_dataset = convert_df_to_ids(
            train_set, self.word2id, self.args.max_seq_length
        )

        dev_dataset = convert_df_to_ids(dev_set, self.word2id, self.args.max_seq_length)

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

        for epoch in tqdm(range(int(self.args.epochs))):
            self.tr_loss = self.train_an_epoch(train_dataloader)[0]
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
        test_dataset = convert_df_to_ids(
            test_set, self.word2id, self.args.max_seq_length
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
        for input_ids, label_ids in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.args.device)
            label_ids = label_ids.to(self.args.device)
            predicted_labels.extend(
                self.model.predict(input_ids).cpu().detach().numpy()
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
        test_dataset = convert_df_to_ids(
            test_set, self.word2id, self.args.max_seq_length
        )

        test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=self.args.batch_size,
        )
        return self.scores(test_dataloader)[0]

    def build_vocab(self, train_df):
        self.intialword2vec()
        self.word2id = {}
        self.embedding_list = []
        for idx, row in train_df.iterrows():
            text = row["docs"].lower()
            for word in text.split(" "):
                if word not in self.word2id and word in self.word2vec:
                    self.word2id[word] = len(self.word2id)
                    self.embedding_list.append(self.word2vec[word])
        del self.word2vec
        self.vocab_size = len(self.word2id)
        self.embedding_list.append([0] * self.embed_dim)  # pad embedding
        print(
            f"Initialize vocab and embedding, vocab_size: {self.vocab_size}, "
            f" embed_dim：{ self.embed_dim}"
        )

    def intialword2vec(self, emb_name="googlenews"):
        emb_path = embedding_path_dict[emb_name]["path"]
        bin_flag = embedding_path_dict[emb_name]["binary"]
        model = gensim.models.KeyedVectors.load_word2vec_format(
            emb_path, binary=bin_flag
        )
        self.embed_dim = len(model.wv["word"])
        self.word2vec = model.wv
