import sys

sys.path.append("../")

from beta_nlp.models.bert_cls import BertModel
from beta_nlp.utils.data_util import biocaser2text
from beta_nlp.utils.textloader import create_examples_from_df


config = {
    "pretrained_model": "bert-base-cased",
    "tokenizer": "bert-base-cased",
    "max_seq_length": 256,
    "batch_size": 16,
    "lr": 2e-5,
    "epochs": 30,
    "device": "cuda",
    "gpu_ids": "3",
    "seed": 2020,
    "fp16": False,
    "loss_scale": 0,
    "warmup_proportion": 0.1,
    "gradient_accumulation_steps": 1,
    "num_labels": 2,
    "is_multilabel": False,
    "valid_metric": "f1",
    "model_save_dir": "../checkpoints/bert_20200902/",
    "patience": 4,
}

cls = BertModel(config)

data_df = biocaser2text("../datasets/biocaster/BioCaster.3.xml")

train_examples = create_examples_from_df(data_df)

cls.train(train_examples, train_examples)
