import html
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

nltk.data.path.append("./nltk_data/")

# todo: Unify the preprocess using spacy: https://spacy.io/.
class BasicTextPreprocess:
    """
    This class contains all text pre-processing function
    # Input parameters: Dataframe, Column_name on which function needs to be applied
    # Output parameters: Return dataframe after applying operations
    """

    # todo: Pass functions as a list of arguments to apply in the class
    # todo: make set of words before applying all operations to reduce processing time.
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatiser = WordNetLemmatizer()

    def process(self, data_df, column_name):
        data_df = self.clean_html(data_df, column_name)
        data_df = self.remove_non_ascii(data_df, column_name)
        data_df = self.remove_spaces(data_df, column_name)
        data_df = self.remove_punctuation(data_df, column_name)
        data_df = self.stemming(data_df, column_name)
        data_df = self.lemmatization(data_df, column_name)
        data_df = self.stop_words(data_df, column_name)
        return data_df

    def remove_non_ascii(self, data_df, column_name):
        data_df.loc[:, column_name] = data_df[column_name].apply(
            lambda x: "".join(i for i in x if ord(i) < 128)
        )
        return data_df

    def clean_html(self, data_df, column_name):
        """remove html entities"""
        print("clean_html...")
        data_df.loc[:, column_name] = data_df[column_name].apply(html.unescape)
        return data_df

    def remove_spaces(self, data_df, column_name):
        print("remove_spaces...")
        data_df.loc[:, column_name] = data_df[column_name].apply(
            lambda x: x.replace("\n", " ")
        )
        data_df.loc[:, column_name] = data_df[column_name].apply(
            lambda x: x.replace("\t", " ")
        )
        data_df.loc[:, column_name] = data_df[column_name].apply(
            lambda x: x.replace("  ", " ")
        )
        data_df.loc[:, column_name] = data_df[column_name].apply(lambda x: x.lower())
        return data_df

    def remove_punctuation(self, data_df, column_name):
        print("remove_punctuation...")
        tr = str.maketrans("", "", string.punctuation)
        # data_df[column_name] = data_df[column_name].apply(lambda x: " ".join([item.translate(tr)
        #                                                                 for item in x.split()]))
        data_df.loc[:, column_name] = data_df[column_name].apply(
            lambda x: x.translate(tr)
        )
        return data_df

    def stemming(self, data_df, column_name):
        print("stemming...")
        # todo: provide option of selecting stemmer.
        snowball_stemmer = SnowballStemmer("english")
        # data_df[column_name] = data_df[column_name].apply(lambda x: " ".join([snowball_stemmer.stem(item)
        #                                                                 for item in x.split()]))
        data_df.loc[:, column_name] = data_df[column_name].apply(
            lambda x: " ".join([self.stemmer.stem(item) for item in x.split()])
        )
        return data_df

    def lemmatization(self, data_df, column_name):
        print("lemmatization...")
        data_df.loc[:, column_name] = data_df[column_name].apply(
            lambda x: " ".join([self.lemmatiser.lemmatize(item) for item in x.split()])
        )
        return data_df

    def stop_words(self, data_df, column_name):
        stop = stopwords.words("english")
        data_df.loc[:, column_name] = data_df[column_name].apply(
            lambda x: " ".join(set([item for item in x.split() if item not in stop]))
        )
        return data_df
