import time

import numpy as np

from ..utils import get_rng


def cls_eval(model, metrics, X, y_true, verbose=False):
    """Evaluate model on provided classification metrics.

    Args:
        model: Classification model
        metrics:  List of rating metrics :obj:`src.metrics.ClsMetric`.
        test_set (DataFrame):  Dataset to be used for evaluation.
        verbose (bool):  Output evaluation progress.

    Returns:
        list: average result for each of the metrics
    """

    if len(metrics) == 0:
        return []

    avg_results = {}
    if "Word2vec" in model.name:
        y_pred = model.predict(X)
    else:
        y_pred = model.predict(np.stack(X))

    for mt in metrics:
        avg_results[mt.name] = mt.compute(y_true, y_pred)
    return avg_results


class BaseEval:
    """Base Evaluation Method"""

    def __init__(self, fmt="UIR", seed=None, verbose=False, **kwargs):
        self.fmt = fmt
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.verbose = verbose
        self.seed = seed
        self.rng = get_rng(seed)

    def _reset(self):
        """Reset the random number generator for reproducibility"""
        self.rng = get_rng(self.seed)
        # self.test_set = self.test_set.reset()

    def _eval(self, model, test_set, metrics, columns):
        X = test_set[columns[0]]
        y_true = test_set[columns[1]]
        metrics_results = cls_eval(model, metrics, X, y_true, verbose=self.verbose)
        # metrics_results["model"] = model.name
        return metrics_results

    def evaluate(
        self, data_df, model, metrics, columns, show_validation=True,
    ):
        """Evaluate given models according to given metrics"""
        if self.train_set is None:
            raise ValueError("train_set is required but None!")
        if self.test_set is None:
            raise ValueError("test_set is required but None!")

        self._reset()

        ###########
        # FITTING #
        ###########
        if self.verbose:
            print("\n[{}] Training started!".format(model.name))

        start = time.time()
        X = list(self.train_set[columns[0]])
        y_true = list(self.train_set[columns[1]])
        model.fit(X, y_true)
        train_time = time.time() - start

        ##############
        # EVALUATION #
        ##############
        if self.verbose:
            print("\n[{}] Evaluation started!".format(model.name))

        start = time.time()
        test_result = self._eval(
            model=model, test_set=self.test_set, metrics=metrics, columns=columns
        )
        test_time = time.time() - start
        test_result["Train (s)"] = train_time
        test_result["Test (s)"] = test_time

        val_result = None
        # if show_validation and self.val_set is not None:
        #     start = time.time()
        #     val_result = self._eval(
        #         model=model, test_set=self.val_set, metrics=metrics, columns=columns
        #     )
        #     val_time = time.time() - start
        #     val_result.metric_avg_results["Time (s)"] = val_time
        # @todo for validation prtocol

        return test_result, val_result

    def evaluate_test(
        self, test_set, model, metrics, columns, show_validation=True,
    ):
        """Evaluate given models according to given metrics"""

        ##############
        # EVALUATION #
        ##############
        if self.verbose:
            print("\n[{}] Evaluation started!".format(model.name))

        start = time.time()
        test_result = self._eval(
            model=model, test_set=test_set, metrics=metrics, columns=columns
        )
        test_time = time.time() - start
        test_result["Test (s)"] = test_time

        val_result = None
        # if show_validation and self.val_set is not None:
        #     start = time.time()
        #     val_result = self._eval(
        #         model=model, test_set=self.val_set, metrics=metrics, columns=columns
        #     )
        #     val_time = time.time() - start
        #     val_result.metric_avg_results["Time (s)"] = val_time
        # @todo for validation prtocol

        return test_result, val_result
