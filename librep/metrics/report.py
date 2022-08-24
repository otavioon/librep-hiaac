from functools import partial
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from librep.base.evaluators import SupervisedEvaluator
from librep.config.type_definitions import ArrayLike

classification_report_dict = partial(classification_report, output_dict=True)


class ClassificationReport(SupervisedEvaluator):
    def __init__(
        self,
        use_accuracy: bool = True,
        use_f1_score: bool = True,
        use_confusion_matrix: bool = True,
        use_classification_report: bool = False,
    ):
        self.use_accuracy = use_accuracy
        self.use_f1_score = use_f1_score
        self.use_confusion_matrix = use_confusion_matrix
        self.use_classification_report = use_classification_report

    def evaluate(
        self, y_true: ArrayLike, y_pred: ArrayLike, **evaluator_params
    ) -> dict:
        result = {}

        if self.use_accuracy:
            res = accuracy_score(y_true, y_pred)
            result["accuracy"] = float(res)

        if self.use_f1_score:
            res = f1_score(y_true, y_pred, average="weighted")
            result["f1 score (weighted)"] = float(res)

            res = f1_score(y_true, y_pred, average="micro")
            result["f1 score (micro)"] = float(res)

            res = f1_score(y_true, y_pred, average="macro")
            result["f1 score (macro)"] = float(res)

        if self.use_confusion_matrix:
            res = confusion_matrix(y_true, y_pred)
            result["confusion matrix"] = res.tolist()

        if self.use_classification_report:
            res = classification_report(y_true, y_pred, output_dict=True)
            result["classification report"] = res

        return result