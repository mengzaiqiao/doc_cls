from tqdm import tqdm

from .base_eval import BaseEval


class TrainDevTest(BaseEval):
    """Cross Validation Evaluation Method."""

    def __init__(self, seed=None, verbose=False, **kwargs):
        BaseEval.__init__(self, seed=seed, verbose=verbose, **kwargs)
        self.seed = seed

    def evaluate(self, data_df, model, metrics, columns, show_validation=False):
        self.train_set = data_df[data_df["flag"] == "train"]
        self.dev_set = data_df[data_df["flag"] == "valid"]
        self.test_set = data_df[data_df["flag"] == "test"]
        test_res = super(TrainDevTest, self).evaluate(
            data_df, model, metrics, columns, False
        )[0]
        return test_res, None
