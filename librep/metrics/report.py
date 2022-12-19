from pathlib import Path

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from librep.base.evaluators import SupervisedEvaluator
from librep.config.type_definitions import ArrayLike, PathLike


class ClassificationReport(SupervisedEvaluator):
    def __init__(
        self,
        use_accuracy: bool = True,
        use_f1_score: bool = True,
        use_confusion_matrix: bool = True,
        use_classification_report: bool = False,
        plot_confusion_matrix: bool = True,
        output_path: PathLike = None,
        # Parameters to confusion matrix
        normalize: str = None,
        display_labels: ArrayLike = None,
        xticks_rotation: str = 'horizontal'
    ):
        self.use_accuracy = use_accuracy
        self.use_f1_score = use_f1_score
        self.use_confusion_matrix = use_confusion_matrix
        self.use_classification_report = use_classification_report
        self.plot_confusion_matrix = plot_confusion_matrix
        self.output_path = Path(output_path) if output_path is not None else None 
        self.normalize = normalize
        self.display_labels = display_labels
        self.xticks_rotation = xticks_rotation

        # TODO Save
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)

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

        if self.plot_confusion_matrix:
            if self.display_labels is not None:
                self.xticks_rotation = 'vertical'
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize=self.normalize,
                                                    display_labels=self.display_labels, 
                                                    xticks_rotation=self.xticks_rotation)
            plt.show()

        return result