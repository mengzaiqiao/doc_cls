import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TextLinear(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def generate_batch_wo_label(batch):
    text = [torch.tensor(entry) for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [torch.tensor(entry[1]) for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


class RandomWord2vec(object):
    def __init__(
        self, data_df, emb_dim=300, batch_size=128, max_epochs=50, num_class=2
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from utils.preprocess import PreProcess

        pre_processor = PreProcess(data_df, "docs", lower=False)
        # todo: change code to provide all functions in class definition.
        pre_processor.clean_html()
        pre_processor.remove_non_ascii()
        pre_processor.remove_spaces()
        pre_processor.remove_punctuation()
        pre_processor.stop_words()
        # pre_processor.tokenize()
        data_df.head()
        self.data_df = data_df
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_class = num_class
        self.build_vocab()
        self.data_df.docs = self.data_df.docs.apply(self.doc2idx)

    def doc2idx(self, doc):
        idxs = []
        if type(doc) is not list:
            doc = doc.split()
        for word in doc:
            idxs.append(self.word_to_ix[word])
        return idxs

    def build_vocab(self):
        word_to_ix = {}
        for idx, row in self.data_df.iterrows():
            doc = row.docs
            if type(doc) is not list:
                doc = doc.split()
            for word in doc:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        self.word_to_ix = word_to_ix
        self.vocab_size = len(self.word_to_ix)

    def train_epoch(self, X, y):

        # Train the model
        train_loss = 0
        train_acc = 0
        tran_data = np.stack((y, X), axis=-1)
        data = DataLoader(
            tran_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=generate_batch,
        )
        for i, (text, offsets, cls) in enumerate(data):
            self.optimizer.zero_grad()
            text, offsets, cls = (
                text.to(self.device),
                offsets.to(self.device),
                cls.to(self.device),
            )
            output = self.model(text, offsets)
            loss = self.criterion(output, cls)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            train_acc += (output.argmax(1) == cls).sum().item()

        # Adjust the learning rate
        self.scheduler.step()

        return train_loss / len(X), train_acc / len(X)

    def fit(self, X, y):
        self.model = TextLinear(self.vocab_size, self.emb_dim, self.num_class).to(
            self.device
        )
        min_valid_loss = float("inf")
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=4.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

        for epoch in range(self.max_epochs):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch(X, y)
            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print(
                "Epoch: %d" % (epoch + 1),
                " | time in %d minutes, %d seconds" % (mins, secs),
            )
            print(
                f"\tLoss: {train_loss:.4f}(train)\t|\tAcc:"
                f" {train_acc * 100:.1f}%(train)"
            )

    def predict(self, X):
        loss = 0
        acc = 0
        data = DataLoader(
            X, batch_size=self.batch_size, collate_fn=generate_batch_wo_label
        )
        pred_y = []
        for text, offsets in data:
            text, offsets = text.to(self.device), offsets.to(self.device)
            with torch.no_grad():
                output = self.model(text, offsets)
                pred_y = output.argmax(1)
        return pred_y

    #     return loss / len(data_), acc / len(data_)

    def test(self, data_):
        loss = 0
        acc = 0
        data = DataLoader(data_, batch_size=self.batch_size, collate_fn=generate_batch)
        pred_y = []
        for text, offsets, cls in data:
            text, offsets, cls = (
                text.to(self.device),
                offsets.to(self.device),
                cls.to(self.device),
            )
            with torch.no_grad():
                output = self.model(text, offsets)
                loss = criterion(output, cls)
                loss += loss.item()
                pred_y = output.argmax(1)
                acc += (output.argmax(1) == cls).sum().item()
        #     return pred_y
        return loss / len(data_), acc / len(data_)
