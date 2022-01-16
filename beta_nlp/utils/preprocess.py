import html
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from torchtext.data.utils import get_tokenizer

nltk.data.path.append("./nltk_data/")

def pad_input_matrix(unpadded_matrix, max_doc_length):
    """
    Returns a zero-padded matrix for a given jagged list
    :param unpadded_matrix: jagged list to be padded
    :return: zero-padded matrix
    """
    max_doc_length = min(max_doc_length, max(len(x) for x in unpadded_matrix))
    zero_padding_array = [0 for i0 in range(len(unpadded_matrix[0][0]))]

    for i0 in range(len(unpadded_matrix)):
        if len(unpadded_matrix[i0]) < max_doc_length:
            unpadded_matrix[i0] += [zero_padding_array for i1 in range(max_doc_length - len(unpadded_matrix[i0]))]
        elif len(unpadded_matrix[i0]) > max_doc_length:
            unpadded_matrix[i0] = unpadded_matrix[i0][:max_doc_length]

def convert_examples_to_features(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = [float(x) for x in example.label]

        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s" % example.label)

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id))
    return features


def convert_examples_to_hierarchical_features(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = [tokenizer.tokenize(line) for line in sent_tokenize(example.text_a)]
        tokens_b = None

        if example.text_b:
            tokens_b = [tokenizer.tokenize(line) for line in sent_tokenize(example.text_b)]
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length
            # Account for [CLS], [SEP], [SEP]
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP]
            for i0 in range(len(tokens_a)):
                if len(tokens_a[i0]) > max_seq_length - 2:
                    tokens_a[i0] = tokens_a[i0][:(max_seq_length - 2)]

        tokens = [["[CLS]"] + line + ["[SEP]"] for line in tokens_a]
        segment_ids = [[0] * len(line) for line in tokens]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = list()
        for line in tokens:
            input_ids.append(tokenizer.convert_tokens_to_ids(line))

        # Input mask has 1 for real tokens and 0 for padding tokens
        input_mask = [[1] * len(line_ids) for line_ids in input_ids]

        # Zero-pad up to the sequence length.
        padding = [[0] * (max_seq_length - len(line_ids)) for line_ids in input_ids]
        for i0 in range(len(input_ids)):
            input_ids[i0] += padding[i0]
            input_mask[i0] += padding[i0]
            segment_ids[i0] += padding[i0]

        label_id = [float(x) for x in example.label]

        if print_examples and ex_index < 5:
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s" % example.label)

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id))
    return featuresF

# todo: Modify class to include BaseEstimator and TransformerMixin to create a sophisticated pipeline.
# https://github.com/pratos/flask_api/blob/master/notebooks/AnalyticsVidhya%20Article%20-%20ML%20Model%20approach.ipynb
class PreProcess:
    """
    This class contains all text pre-processing function
    # Input parameters: Dataframe, Column_name on which function needs to be applied
    # Output parameters: Return dataframe after applying operations
    """

    # todo: Pass functions as a list of arguments to apply in the class
    # todo: make set of words before applying all operations to reduce processing time.
    def __init__(self, data, column_name, lower=True):
        self.data = data
        self.column_name = column_name
        self.stemmer = PorterStemmer()
        self.lemmatiser = WordNetLemmatizer()
        if lower:
            self.data[self.column_name] = self.data[self.column_name].apply(
                lambda x: x.lower()
            )
        # pass\

    def tokenize(self):
        tokenizer = get_tokenizer("basic_english")
        self.data[self.column_name] = self.data[self.column_name].apply(tokenizer)

    def remove_non_ascii(self):
        self.data[self.column_name] = self.data[self.column_name].apply(
            lambda x: "".join(i for i in x if ord(i) < 128)
        )
        return self.data

    def clean_html(self):
        """remove html entities"""
        self.data[self.column_name] = self.data[self.column_name].apply(html.unescape)
        return self.data

    def remove_spaces(self):
        self.data[self.column_name] = self.data[self.column_name].apply(
            lambda x: x.replace("\n", " ")
        )
        self.data[self.column_name] = self.data[self.column_name].apply(
            lambda x: x.replace("\t", " ")
        )
        self.data[self.column_name] = self.data[self.column_name].apply(
            lambda x: x.replace("  ", " ")
        )
        return self.data

    def remove_punctuation(self):
        tr = str.maketrans("", "", string.punctuation)
        # self.data[self.column_name] = self.data[self.column_name].apply(lambda x: " ".join([item.translate(tr)
        #                                                                 for item in x.split()]))
        self.data[self.column_name] = self.data[self.column_name].apply(
            lambda x: x.translate(tr)
        )
        return self.data

    def stemming(self):
        # todo: provide option of selecting stemmer.
        snowball_stemmer = SnowballStemmer("english")
        # self.data[self.column_name] = self.data[self.column_name].apply(lambda x: " ".join([snowball_stemmer.stem(item)
        #                                                                 for item in x.split()]))
        self.data[self.column_name] = self.data[self.column_name].apply(
            lambda x: " ".join([self.stemmer.stem(item) for item in x.split()])
        )
        return self.data

    def lemmatization(self):
        self.data[self.column_name] = self.data[self.column_name].apply(
            lambda x: " ".join([self.lemmatiser.lemmatize(item) for item in x.split()])
        )
        return self.data

    def stop_words(self):
        stop = stopwords.words("english")
        self.data[self.column_name] = self.data[self.column_name].apply(
            lambda x: " ".join(set([item for item in x.split() if item not in stop]))
        )
        return self.data
