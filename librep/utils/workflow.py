from typing import List

from librep.base.transform import Transform
from librep.base.data import Dataset
from librep.base.estimator import Estimator
from librep.config.type_definitions import ArrayLike
from librep.utils.transform import TransformDataset

class SimpleTrainEvalWorkflow:
    def __init__(self, estimator: Estimator, transforms: List[Transform] = None, is_supervised: bool = True, evaluate: bool = True, evaluator: callable = None):
        self.transforms = transforms
        self.estimator = estimator
        self.is_supervised = is_supervised
        self.evaluate = evaluate
        self.evaluator = evaluator

    def __call__(self, train_dataset: Dataset, test_dataset: Dataset = None):
        if self.transforms is not None:
            transformer = TransformDataset(transforms=self.transforms)
            train_dataset = transformer(train_dataset)
            if test_dataset is not None:
                test_dataset = transformer(test_dataset)

        if self.is_supervised:
            self.estimator.fit(train_dataset[:][0], y=train_dataset[:][1])
        else:
            self.estimator.fit(train_dataset[:][0])

        if self.evaluate:
            if test_dataset:
                y_pred = self.estimator.predict(test_dataset[:][0])
            else:
                y_pred = self.estimator.predict(train_dataset[:][0])

        else:
            y_pred = []

        if self.evaluator is not None:
            if test_dataset:
                return self.evaluator(y_pred, test_dataset[:][1])
            else:
                return self.evaluator(y_pred, train_dataset[:][1])
        else:
            return y_pred
