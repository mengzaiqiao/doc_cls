from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ClsMetric:
    """Classification Metric.
    Args:
        type (string): value: 'classification'. Type of the metric, e.g., "ranking", "rating", "classification"
        name (string): default: None
            Name of the measure.
        k (int): optional, default: 2
            The number of classes
    """

    def __init__(self, name=None, k=2):
        self.type = "classification"
        self.name = name
        self.k = k

    def compute(self, **kwargs):
        raise NotImplementedError()


class Precision(ClsMetric):
    """Compute the precision"""

    def __init__(self, k=2):
        ClsMetric.__init__(self, name="precision", k=k)

    def compute(self, y_true, y_pred, average="binary", **kwargs):
        """Compute the precision score.

        Args:
            y_true (Numpy array):
                Ground-truth label values.
            y_pred (Numpy array):
                Predicted label values.
            average (string), [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                This parameter is required for multiclass/multilabel targets. If None, the scores for each class are
                returned. Otherwise, this determines the type of averaging performed on the data:
                'binary':
                Only report results for the class specified by pos_label. This is applicable only if targets
                (y_{true,pred}) are binary.

                'micro':
                Calculate metrics globally by counting the total true positives, false negatives and false positives.

                'macro':
                Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance
                into account.

                'weighted':
                Calculate metrics for each label, and find their average weighted by support (the number of true
                instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an
                F-score that is not between precision and recall.

                'samples':
                Calculate metrics for each instance, and find their average (only meaningful for multilabel
                classification where this differs from accuracy_score).
            **kwargs: For compatibility

        Returns
            float: precision score.
        """
        if self.k > 2:
            average = "macro"
        score = precision_score(y_true, y_pred, average=average)
        return score


class Accuracy(ClsMetric):
    """Compute the accuracy score."""

    def __init__(self, k=2):
        ClsMetric.__init__(self, name="accuracy")

    def compute(self, y_true, y_pred, **kwargs):
        """Compute the accuracy score.

        Args:
            y_true (Numpy array):
                Ground-truth label values.
            y_pred (Numpy array):
                Predicted label values.
            **kwargs: For compatibility
        Returns
            float: accuracy score.
        """
        score = accuracy_score(y_true, y_pred)
        return score


class Recall(ClsMetric):
    """Compute the recall score."""

    def __init__(self, k=2):
        ClsMetric.__init__(self, name="recall", k=k)

    def compute(self, y_true, y_pred, average="binary", **kwargs):
        """Compute the recall score.

        Args:
            y_true (Numpy array):
                Ground-truth label values.
            y_pred (Numpy array):
                Predicted label values.
            average (string), [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                This parameter is required for multiclass/multilabel targets. If None, the scores for each class are
                returned. Otherwise, this determines the type of averaging performed on the data:
                'binary':
                Only report results for the class specified by pos_label. This is applicable only if targets
                (y_{true,pred}) are binary.

                'micro':
                Calculate metrics globally by counting the total true positives, false negatives and false positives.

                'macro':
                Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance
                into account.

                'weighted':
                Calculate metrics for each label, and find their average weighted by support (the number of true
                instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an
                F-score that is not between precision and recall.

                'samples':
                Calculate metrics for each instance, and find their average (only meaningful for multilabel
                classification where this differs from accuracy_score).
            **kwargs: For compatibility
        Returns
            float: recall score.
        """
        if self.k > 2:
            average = "macro"
        score = recall_score(y_true, y_pred, average=average)
        return score


class F1Score(ClsMetric):
    """Compute the F1 score, also known as balanced F-score or F-measure."""

    def __init__(self, k=2):
        ClsMetric.__init__(self, name="f1_score", k=k)

    def compute(self, y_true, y_pred, average="binary", **kwargs):
        """Compute the recall score.

        Args:
            y_true (Numpy array):
                Ground-truth label values.
            y_pred (Numpy array):
                Predicted label values.
            average (string), [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                This parameter is required for multiclass/multilabel targets. If None, the scores for each class are
                returned. Otherwise, this determines the type of averaging performed on the data:
                'binary':
                Only report results for the class specified by pos_label. This is applicable only if targets
                (y_{true,pred}) are binary.

                'micro':
                Calculate metrics globally by counting the total true positives, false negatives and false positives.

                'macro':
                Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance
                into account.

                'weighted':
                Calculate metrics for each label, and find their average weighted by support (the number of true
                instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an
                F-score that is not between precision and recall.

                'samples':
                Calculate metrics for each instance, and find their average (only meaningful for multilabel
                classification where this differs from accuracy_score).
            **kwargs: For compatibility
        Returns
            float: f1_score score.
        """
        if self.k > 2:
            average = "macro"
        score = f1_score(y_true, y_pred, average=average)
        return score


class RocAuc(ClsMetric):
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.."""

    def __init__(self, k=2):
        ClsMetric.__init__(self, name="auc", k=k)

    def compute(self, y_true, y_pred, average="binary", **kwargs):
        """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores..

        Args:
            y_true (Numpy array):
                Ground-truth label values.
            y_pred (Numpy array):
                Predicted label values.
            average (string), [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
                This parameter is required for multiclass/multilabel targets. If None, the scores for each class are
                returned. Otherwise, this determines the type of averaging performed on the data:
                'binary':
                Only report results for the class specified by pos_label. This is applicable only if targets
                (y_{true,pred}) are binary.

                'micro':
                Calculate metrics globally by counting the total true positives, false negatives and false positives.

                'macro':
                Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance
                into account.

                'weighted':
                Calculate metrics for each label, and find their average weighted by support (the number of true
                instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an
                F-score that is not between precision and recall.

                'samples':
                Calculate metrics for each instance, and find their average (only meaningful for multilabel
                classification where this differs from accuracy_score).
            **kwargs: For compatibility
        Returns
            float: f1_score score.
        """
        if self.k > 2:
            average = "macro"
        score = roc_auc_score(y_true, y_pred, average=average)
        return score
