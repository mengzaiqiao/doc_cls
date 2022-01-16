import xml.dom.minidom
from os import listdir
from os.path import isfile, join

import pandas as pd

from torchtext.data.utils import get_tokenizer

promed_data_path = "/home/zm324/workspace/doc_cls/datasets/promed_data/"
extended_promed_data_path = "/home/zm324/workspace/doc_cls/datasets/promed_extended/"


def get_raw_df_data(data_path, filter_in=None, filter_out=None):
    docfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    print(f"found {len(docfiles)} files")

    if filter_in:
        docfiles = [f for f in docfiles if filter_in in f]
        print(f"Remain {len(docfiles)} files after filter_in with {filter_in}")

    if filter_out:
        docfiles = [f for f in docfiles if filter_out not in f]
        print(f"Remain {len(docfiles)} files after filter_out with {filter_out}")

    print(f"found files: {len(docfiles)}")
    docs = []
    file_names = []
    for doc_file in docfiles:
        file_names.append(doc_file)
        file = data_path + doc_file
        import codecs

        with codecs.open(file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            doc_string = " \n ".join(lines)
        docs.append(doc_string)
    return pd.DataFrame({"docs": docs, "file_name": file_names})


def split_data(data_df, seed=2020):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    from sklearn.utils import shuffle

    data_df = shuffle(data_df).reset_index()
    data_df["flag"] = None
    train_index = int(len(data_df.index) * 0.8)
    valid_index = train_index + round(len(data_df.index) * 0.1)
    data_df.loc[:train_index, "flag"] = "train"
    data_df.loc[train_index:valid_index, "flag"] = "valid"
    data_df.loc[valid_index:, "flag"] = "test"
    train_set = data_df[data_df["flag"] == "train"]
    dev_set = data_df[data_df["flag"] == "valid"]
    test_set = data_df[data_df["flag"] == "test"]
    import time

    # re-initialize seed with a random number (current time)
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    return train_set, dev_set, test_set


def get_raw_promed_data(data_path, filter_in=None, filter_out=None):
    docfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    print(f"found {len(docfiles)} files")

    if filter_in:
        docfiles = [f for f in docfiles if filter_in in f]
        print(f"Remain {len(docfiles)} files after filter_in with {filter_in}")

    if filter_out:
        docfiles = [f for f in docfiles if filter_out not in f]
        print(f"Remain {len(docfiles)} files after filter_out with {filter_out}")

    print(f"found files: {len(docfiles)}")
    docs = []
    file_names = []
    for doc_file in docfiles:
        file_names.append(doc_file)
        file = data_path + doc_file
        import codecs

        with codecs.open(file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            doc_string = " \n ".join(lines)
        docs.append(doc_string)
    return docs


def get_raw_promed_df():
    pos_path = promed_data_path + "pos/"
    neg_path = promed_data_path + "neg/"
    pos_docs = get_raw_promed_data(pos_path)
    neg_docs = get_raw_promed_data(neg_path)
    pos_labels = [1] * len(pos_docs)
    neg_labels = [0] * len(neg_docs)
    return pd.DataFrame(
        {"docs": pos_docs + neg_docs, "labels": pos_labels + neg_labels}
    )


def get_raw_extended_promed_df():
    pos_path = extended_promed_data_path + "pos/"
    neg_path = extended_promed_data_path + "neg/"
    pos_docs = get_raw_promed_data(pos_path, filter_out="alerting")
    alerting_docs = get_raw_promed_data(pos_path, filter_in="alerting")
    neg_docs = get_raw_promed_data(neg_path)
    pos_labels = [1] * len(pos_docs)
    neg_labels = [0] * len(neg_docs)
    alert_labels = [2] * len(alerting_docs)
    return pd.DataFrame(
        {
            "docs": pos_docs + neg_docs + alerting_docs,
            "labels": pos_labels + neg_labels + alert_labels,
        }
    )


def get_promed_data(data_path, stopwords=True, tokenizer=None, lower=True):
    docfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    print(f"found files: {len(docfiles)}")

    if stopwords:
        from nltk.corpus import stopwords

        stopwords = stopwords.words("english")

    if not tokenizer:
        tokenizer = get_tokenizer("basic_english")

    docs = []
    for doc_file in docfiles:
        file = data_path + doc_file
        with open(file) as f:
            lines = f.readlines()
            doc_string = " \n ".join(lines)
            if lower:
                doc_string = doc_string.lower()  # use only lowercase
            #             print(doc_string)
            tokens = tokenizer(doc_string)
            if stopwords:
                tokens = [w for w in tokens if (w not in stopwords) and (w.isalnum())]
        docs.append(tokens)
    return docs


def biocaser2text(data_file):
    docs = []
    labels = []
    ids = []
    with open(data_file) as f:
        text_root = "<root>" + f.read() + "</root>".replace("NAMe", "NAME")
        DOMTree = xml.dom.minidom.parseString(text_root)
        collection = DOMTree.documentElement
        docElements = collection.getElementsByTagName("DOC")
        for docE in docElements:
            doc_string = ""
            label = docE.getAttribute("relevancy")
            #             doc_id = docE.getAttribute("id")
            if not label:
                print(docE.tagName)
            for curNode in docE.childNodes:
                if curNode.hasChildNodes():
                    for curNode_child in curNode.childNodes:
                        if curNode_child.hasChildNodes():
                            for curNode_child_child in curNode_child.childNodes:
                                doc_string += curNode_child_child.data
                                if curNode_child_child.hasChildNodes():
                                    raise ValueError(
                                        "Error!!! Shuld have more hasChildNodes"
                                    )
                        else:
                            doc_string += curNode_child.data
                else:
                    doc_string += curNode.data
            docs.append(doc_string.replace("\n\n", "\n"))
            labels.append(label)
            #             ids.append(doc_id)
            if not label:
                print(doc_string)
            if len(doc_string) < 10 or docE.getAttribute("DOC"):
                print(doc_string)
    print(
        f"parse biocaser data from {data_file}, docs number:{len(docs)}, lablels"
        f" number:{len(labels)}"
    )
    label_dic = {
        "reject": 0,
        "publish": 1,
        "alert": 1,
        "check": 1,
    }
    labels = [label_dic[label] for label in labels]
    df = pd.DataFrame(list(zip(docs, labels)), columns=["docs", "labels"])
    return df


def biocaster2df(data_file):
    """Load a DataFrame format dataset of biocaster

    Args:
        data_file: original data file path

    Returns:
        DataFrame: the loaded dataframe dataset of biocaster
    """
    docs = []
    labels = []
    ids = []
    with open(data_file) as f:
        text_root = "<root>" + f.read() + "</root>".replace("NAMe", "NAME")
        DOMTree = xml.dom.minidom.parseString(text_root)
        collection = DOMTree.documentElement
        docElements = collection.getElementsByTagName("DOC")
        for docE in docElements:
            doc_string = ""
            label = docE.getAttribute("relevancy")
            #             doc_id = docE.getAttribute("id")
            if not label:
                print(docE.tagName)
            for curNode in docE.childNodes:
                if curNode.hasChildNodes():
                    for curNode_child in curNode.childNodes:
                        if curNode_child.hasChildNodes():
                            for curNode_child_child in curNode_child.childNodes:
                                doc_string += curNode_child_child.data
                                if curNode_child_child.hasChildNodes():
                                    raise ValueError(
                                        "Error!!! Shuld have more hasChildNodes"
                                    )
                        else:
                            doc_string += curNode_child.data
                else:
                    doc_string += curNode.data
            docs.append(doc_string.replace("\n\n", "\n"))
            labels.append(label)
            #             ids.append(doc_id)
            if not label:
                print(doc_string)
            if len(doc_string) < 10 or docE.getAttribute("DOC"):
                print(doc_string)
    print(
        f"parse biocaser data from {data_file}, docs number:{len(docs)}, lablels"
        f" number:{len(labels)}"
    )
    label_dic = {
        "reject": 0,
        "publish": 3,
        "alert": 2,
        "check": 1,
    }
    labels = [label_dic[label] for label in labels]
    df = pd.DataFrame(list(zip(docs, labels)), columns=["docs", "labels"])

    return df
