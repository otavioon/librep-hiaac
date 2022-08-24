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
        self, train_dataset: MultiModalDataset, test_datasets: List[MultiModalDataset] = None
    ):
        if test_datasets is not None and not isinstance(test_datasets, list):
            test_datasets = [test_datasets]

        if self.transformer is not None:
            train_dataset = self.transformer(train_dataset)
            if test_datasets is not None:
                test_datasets = [self.transformer(d) for d in test_datasets] 

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
            if test_datasets is not None:
                y_preds = [estimator.predict(d[:][0]) for d in test_datasets]
            else:
                y_preds = [estimator.predict(train_dataset[:][0])]

        else:
            y_preds = []

        if self.evaluator is not None:
            if test_datasets is not None:
                return [
                    self.evaluator.evaluate(y_pred, d[:][1])
                    for y_pred, d in zip(y_preds, test_datasets)
                ]
            else:
                return [
                    self.evaluator.evaluate(y_preds[0], train_dataset[:][1])
                ]
        else:
            return y_preds


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
        self, train_dataset: MultiModalDataset, test_datasets: List[MultiModalDataset] = None
    ):
        runs = []
        for i in range(self.num_runs):
            if self.debug:
                print(f"----- Starting run {i+1} / {self.num_runs} ------")
            start = time.time()
            result = self.workflow(train_dataset, test_datasets)
            end = time.time()
            if self.debug:
                print(result)
                print(f"----- Finished run {i+1} / {self.num_runs}. It took: {end-start:.3f} seconds -----\n")
            runs.append({
                "run id": i+1,
                "start": start,
                "end": end,
                "time taken": end-start,
                "result": result
            })
        return {
            "runs": runs
        }