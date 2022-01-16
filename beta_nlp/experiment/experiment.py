import pandas as pd

from beta_nlp.metrics import ClsMetric

from ..utils.common import save_to_csv
from .result import CVExperimentResult, ExperimentResult


class Experiment:
    """Experiment Class

    Parameters
    ----------
    eval_method: :obj:`<cornac.eval_methods.BaseMethod>`, required
        The evaluation method (e.g., RatioSplit).

    models: array of :obj:`<cornac.models.Recommender>`, required
        A collection of recommender models to evaluate, e.g., [C2PF, HPF, PMF].

    metrics: array of :obj:{`<cornac.metrics.RatingMetric>`, `<cornac.metrics.RankingMetric>`}, required
        A collection of metrics to use to evaluate the recommender models, \
        e.g., [NDCG, MRR, Recall].

    user_based: bool, optional, default: True
        This parameter is only useful if you are considering rating metrics. When True, first the average performance \
        for every user is computed, then the obtained values are averaged to return the final result.
        If `False`, results will be averaged over the number of ratings.

    show_validation: bool, optional, default: True
        Whether to show the results on validation set (if exists).

    save_dir: str, optional, default: None
        Path to a directory for storing trained models and logs. If None,
        models will NOT be stored and logs will be saved in the current working directory.

    Attributes
    ----------
    result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment
        on the test set, initially it is set to None.

    val_result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment
        on the validation set (if exists), initially it is set to None.

    """

    def __init__(
        self,
        data_df,
        preprocessor,
        extractor,
        eval_method,
        models,
        metrics,
        show_validation=True,
        verbose=False,
        result_file=None,
        labels=None,
    ):
        self.data_df = data_df
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.eval_method = eval_method
        self.models = self._validate_models(models)
        self.metrics = self._validate_metrics(metrics)
        self.show_validation = show_validation
        self.verbose = verbose
        self.result_file = result_file
        self.test_result = []
        self.val_result = []
        self.labels = labels

    @staticmethod
    def _validate_models(input_models):
        if not hasattr(input_models, "__len__"):
            raise ValueError(
                "models have to be an array but {}".format(type(input_models))
            )

        valid_models = []
        for model in input_models:
            valid_models.append(model)
        return valid_models

    @staticmethod
    def _validate_metrics(input_metrics):
        if not hasattr(input_metrics, "__len__"):
            raise ValueError(
                "metrics have to be an array but {}".format(type(input_metrics))
            )

        valid_metrics = []
        for metric in input_metrics:
            if isinstance(metric, ClsMetric):
                valid_metrics.append(metric)
        return valid_metrics

    def _create_result(self):
        from ..eval_method.cross_validation import CrossValidation

        if isinstance(self.eval_method, CrossValidation):
            self.test_result = CVExperimentResult()
        else:
            self.test_result = ExperimentResult()
            if self.show_validation and self.eval_method.val_set is not None:
                self.val_result = ExperimentResult()

    def run(self):
        self._create_result()
        if self.preprocessor:
            self.preprocessor.process(self.data_df, "docs")
        if self.extractor:
            self.extractor.extract(self.data_df, "docs")
            feature_columns = self.extractor.feature_columns
        else:
            feature_columns = ["docs"]
        if self.labels:
            data_test = self.data_df[self.data_df["labels"] == 2].copy()
            data_test["test_labels"] = 1
            data_test.name = "promed_alerting"

            dataset_name = getattr(self.data_df, "name", "Default_name")
            self.data_df = self.data_df[self.data_df["labels"].isin(self.labels)]
            self.data_df.name = dataset_name
        print(f"Found feature_columns: {feature_columns}")
        for model in self.models:
            for col in feature_columns:
                print(f"Experiment with {model.name} model on {col} feature.")
                test_result, val_result = self.eval_method.evaluate(
                    self.data_df,
                    model=model,
                    metrics=self.metrics,
                    columns=[col, "labels"],
                )
                test_result["dataset"] = getattr(self.data_df, "name", "Default_name")
                test_result["model"] = model.name
                test_result["feature"] = col
                self.test_result.append(test_result)
                if val_result is not None:
                    val_result["model"] = model.name
                    val_result["feature"] = col
                    self.val_result.append(val_result)

                test_result_df = pd.DataFrame([test_result])
                save_to_csv(test_result_df, self.result_file)
                if self.labels:
                    self.test(data_test, model, col)

    def test(self, data_test, model, col):
        print(f"Evaluate with {model.name} model on {col} feature.")
        test_result, val_result = self.eval_method.evaluate_test(
            data_test, model=model, metrics=self.metrics, columns=[col, "test_labels"],
        )
        test_result["dataset"] = getattr(data_test, "name", "Default_name")
        test_result["model"] = model.name
        test_result["feature"] = col
        self.test_result.append(test_result)
        if val_result is not None:
            val_result["model"] = model.name
            val_result["feature"] = col
            self.val_result.append(val_result)

        test_result_df = pd.DataFrame([test_result])
        save_to_csv(test_result_df, self.result_file)
