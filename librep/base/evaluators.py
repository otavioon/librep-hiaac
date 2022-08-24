from typing import List, Union

from librep.base.estimator import Estimator
from librep.config.type_definitions import ArrayLike


class SupervisedEvaluator:
    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike, **evaluator_params) -> dict:
        raise NotImplementedError


class UnsupervisedEvaluator:
    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike, **evaluator_params) -> dict:
        raise NotImplementedError


class CustomSimpleEvaluator:
    def evalutate(self, model: Estimator, X: ArrayLike, y: ArrayLike) -> dict:
        raise NotImplementedError


class CustomMultiEvaluator:
    def evalutate(self, model: Estimator, Xs: List[ArrayLike], y: List[ArrayLike]) -> dict:
        raise NotImplementedError


Evaluators = Union[
    SupervisedEvaluator,
    UnsupervisedEvaluator,
    CustomSimpleEvaluator,
    CustomMultiEvaluator
]