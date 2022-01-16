from beta_nlp.core.train_engine import TrainEngine
from munch import munchify


class DocCls(TrainEngine):
    def __init__(self, config, name=""):
        self.name = name
        config = munchify(config)
        super(DocCls, self).__init__(config)

    def train(self, train_df, valid_df=None, test_df=None):
        """Train a model, need to be implement for each model.
        If valid_df is specified, the training will be validated by valid_df, otherwise,
        the early stop criterion will be validated by train_df.
        Both valid_df and test_df will be tested for analysis.

        Args:
            train_df:
            valid_df:
            test_df:
        Returns:
        """
        pass

    def test(self, test_df):
        """Score and Evaluate for a dataframe data
        Args:
            test_df:
        Returns:
        """
        self.eval_engine.test_eval(test_df, self.engine.model)

    def load(self, model_dir):
        """Load a trained model
        Args:
            model_dir:
        Returns:
        """
        self.engine.resume_checkpoint(model_dir)

    def predict(self, data_df):
        """
        Args:
            data_df:
        Returns:
            numpy array: the predict score for the user-item pairs
        """
        return self.eval_engine.predict(data_df, self.engine.model)
