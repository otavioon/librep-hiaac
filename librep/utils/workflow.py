from dataclasses import dataclass
from typing import List, Union
import time
import warnings

from librep.base.transform import Transform
from librep.base.data import Dataset
from librep.base.estimator import Estimator
from librep.base.evaluators import Evaluators
from librep.config.type_definitions import ArrayLike
from librep.datasets.common import TransformMultiModalDataset, MultiModalDataset


@dataclass
class NamedDataset:
    name: str
    dataset: Dataset


@dataclass
class NamedEstimator:
    name: str
    estimator: Estimator


@dataclass
class NamedTransform:
    name: str
    transform: Transform


@dataclass
class NamedEvaluator:
    name: str
    evaluator: Evaluators


class SimpleTrainEvalWorkflow:
    def __init__(
        self,
        estimator: Union[type, Estimator],
        estimator_creation_kwags: dict = None,
        do_not_instantiate: bool = False,
        transformer: TransformMultiModalDataset = None,
        do_fit: bool = True,
        is_supervised: bool = True,
        evaluate: bool = True,
        evaluator: Evaluators = None,
        debug: bool = False,
    ):
        self.transformer = transformer
        self.estimator = estimator
        self.estimator_creation_kwags = estimator_creation_kwags or {}
        self.do_not_instantiate = do_not_instantiate
        self.do_fit = do_fit
        self.is_supervised = is_supervised
        self.evaluate = evaluate
        self.evaluator = evaluator
        self.debug = debug

    def __call__(
        self, train_dataset: MultiModalDataset, test_dataset: MultiModalDataset = None
    ):
        if self.transformer is not None:
            train_dataset = self.transformer(train_dataset)
            if test_dataset is not None:
                test_dataset = self.transformer(test_dataset)

        if self.do_not_instantiate:
            estimator = self.estimator
        else:
            estimator = self.estimator(**self.estimator_creation_kwags)

        if self.do_fit:
            if self.is_supervised:
                estimator.fit(train_dataset[:][0], y=train_dataset[:][1])
            else:
                estimator.fit(train_dataset[:][0])

        if self.evaluate:
            if test_dataset:
                y_pred = estimator.predict(test_dataset[:][0])
            else:
                y_pred = estimator.predict(train_dataset[:][0])

        else:
            y_pred = []

        if self.evaluator is not None:
            if test_dataset:
                return self.evaluator.evaluate(y_pred, test_dataset[:][1])
            else:
                return self.evaluator.evaluate(y_pred, train_dataset[:][1])
        else:
            return y_pred


class MultiRunWorkflow:
    def __init__(
        self,
        workflow: callable,
        num_runs: int = 1,
        debug: bool = False,
    ):
        self.workflow = workflow
        self.num_runs = num_runs
        self.debug = debug

    def __call__(
        self, train_dataset: MultiModalDataset, test_dataset: MultiModalDataset = None
    ):
        runs = []
        for i in range(self.num_runs):
            if self.debug:
                print(f"----- Starting run {i} ------")
            start = time.time()
            result = self.workflow(train_dataset, test_dataset)
            end = time.time()
            if self.debug:
                print(result)
                print(f"----- Finished run {i}. It took: {end-start:.3f} seconds -----\n")
            runs.append({
                "id": i,
                "start": start,
                "end": end,
                "time taken": end-start,
                "result": result
            })
        return {
            "runs": runs
        }