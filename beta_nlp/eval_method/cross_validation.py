from sklearn.model_selection import RepeatedKFold
from tqdm import tqdm

from .base_eval import BaseEval


class CrossValidation(BaseEval):
    """Cross Validation Evaluation Method."""

    def __init__(self, n_splits=10, n_repeats=10, seed=None, verbose=False, **kwargs):
        BaseEval.__init__(self, seed=seed, verbose=verbose, **kwargs)
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.seed = seed

    def evaluate(self, data_df, model, metrics, columns, show_validation=False):
        rkf = RepeatedKFold(
            n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.seed
        )
        result_dic_li = []
        split_num = 0
        for train_index, test_index in tqdm(rkf.split(data_df)):
            self.train_set = data_df.iloc[train_index]
            self.test_set = data_df.iloc[test_index]
            test_res = super(CrossValidation, self).evaluate(
                data_df, model, metrics, columns, False
            )[0]
            result_dic_li.append(test_res)
            split_num += 1
        print(f"total num of splits: {split_num}")
        test_result = {}
        for resut_dic in result_dic_li:
            for k, v in resut_dic.items():
                if k not in test_result:
                    test_result[k] = v / split_num
                else:
                    test_result[k] = test_result[k] + (v / split_num)
        return test_result, None
