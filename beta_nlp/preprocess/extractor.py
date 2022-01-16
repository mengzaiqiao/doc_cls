import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class BasicTextFeatureExtraction:
    def __init__(self, feature_list):
        """BasicTextFeatureExtraction

        Args:
            feature_list: A list with elements being string or list
        """
        assert isinstance(feature_list, list)
        self.feature_list = feature_list
        self.feature_columns = []

    def extract(self, data_df, column_name):
        for feature in self.feature_list:
            if not isinstance(feature, list):
                extractor = getattr(self, feature)
                extractor(data_df, column_name)
                if feature not in self.feature_columns:
                    self.feature_columns.append(feature)
            else:
                feature_str = "+".join(feature)
                if feature_str in data_df.columns:
                    continue
                sub_feature_li = []
                for sub_feature in feature:
                    if sub_feature not in data_df.columns:
                        extractor = getattr(self, sub_feature)
                        extractor(data_df, column_name)
                    sub_feature_li.append(np.array(data_df[sub_feature].tolist()))
                data_df.insert(
                    len(data_df.columns),
                    feature_str,
                    list(np.concatenate(sub_feature_li, axis=1)),
                )
                if feature_str not in self.feature_columns:
                    self.feature_columns.append(feature_str)
        return data_df

    def bow(self, data_df, column_name):
        """Bag of Words

        Args:
            data_df:
            column_name:

        Returns:

        """
        vectorizer = CountVectorizer()
        data_df["bow"] = list(
            vectorizer.fit_transform(data_df[column_name].to_numpy()).toarray()
        )

    def tfidf(self, data_df, column_name, min_df=1):
        """Extract tfidf feature

        Args:
            data_df:
            column_name:
            min_df:

        Returns:

        """
        vectorizer = TfidfVectorizer(min_df=min_df)
        data_df["tfidf"] = list(
            vectorizer.fit_transform(data_df[column_name].to_numpy()).toarray()
        )

    def bigram(self, data_df, column_name):
        """Extract Bi-gram feature

        Args:
            data_df:
            column_name:

        Returns:

        """
        vectorizer = CountVectorizer(ngram_range=(2, 2), min_df=2)
        data_df["bigram"] = list(
            vectorizer.fit_transform(data_df[column_name].to_numpy()).toarray()
        )

    def trigram(self, data_df, column_name):
        """Extract Tri-gram feature

        Args:
            data_df:
            column_name:

        Returns:

        """
        vectorizer = CountVectorizer(ngram_range=(3, 3), min_df=2)
        data_df["trigram"] = list(
            vectorizer.fit_transform(data_df[column_name].to_numpy()).toarray()
        )
