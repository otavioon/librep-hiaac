import importlib
import inspect
import pkgutil
import yaml
from functools import partial

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Set

from librep.base.data import Dataset
from librep.base.estimator import Estimator
from librep.base.transform import Transform
from librep.base.evaluators import Evaluators
from librep.config.type_definitions import PathLike


import librep.transforms.fft
import librep.transforms.autocorrelation
import librep.transforms.resampler
import librep.transforms.spectrogram
import librep.transforms.stats
import librep.transforms.tsne
import librep.transforms.umap
import librep.estimators
import librep.metrics.report


class InvalidOperatorType(Exception):
    pass


class ObjectCreator:
    def __init__(
        self,
        transform_lib: str = "librep.transforms",
        estimator_lib: str = "librep.estimators",
        evaluator_lib: str = "librep.evaluators",
    ):
        self.transform_classes = {
            "FFT": librep.transforms.fft.FFT,
            "AutoCorrelation": librep.transforms.autocorrelation.AutoCorrelation,
            "SimpleResampler": librep.transforms.resampler.SimpleResampler,
            "Spectrogram": librep.transforms.spectrogram.Spectrogram,
            "StatsTransform": librep.transforms.stats.StatsTransform,
            "TSNE": librep.transforms.tsne.TSNE,
            "UMAP": librep.transforms.umap.UMAP,
        }

        self.estimator_classes = {
            "KNeighborsClassifier": librep.estimators.KNeighborsClassifier,
            "RandomForestClassifier": librep.estimators.RandomForestClassifier,
            "SVC": librep.estimators.SVC,
        }

        self.evaluator_classes = {
            "ClassificationReport": librep.metrics.report.ClassificationReport
        }

    def create_transform(self, name: str, *args, **kwargs) -> Transform:
        return self.transform_classes[name](*args, **kwargs)

    def create_estimator(self, name: str, *args, **kwargs) -> Estimator:
        return self.estimator_classes[name](*args, **kwargs)

    def create_evaluator(self, name: str, *args, **kwargs) -> Evaluators:
        return self.evaluator_classes[name](*args, **kwargs)


class OperatorCreator:
    def __init__(self, database: dict):
        self._database = database

    def create(self, name: str, *args, **kwargs):
        return self._database[name](*args, **kwargs)

    @staticmethod
    def from_yaml(path: PathLike, creator: ObjectCreator = None) -> "OperatorCreator":
        path = Path(path)
        creator = creator or ObjectCreator()

        with path.open("r") as f:
            values = yaml.load(f, Loader=yaml.FullLoader)

        if "operators" not in values:
            return OperatorCreator(dict())

        database = dict()
        for op_name, op_values in values["operators"].items():
            args = op_values.get("args", tuple())
            kwargs = op_values.get("kwargs", dict())
            if "transform" in op_values:
                name = op_values["transform"]
                database[op_name] = partial(
                    creator.create_transform, name, *args, **kwargs
                )
            elif "estimator" in op_values:
                name = op_values["estimator"]
                database[op_name] = partial(
                    creator.create_estimator, name, *args, **kwargs
                )
            elif "evaluator" in op_values:
                name = op_values["evaluator"]
                database[op_name] = partial(
                    creator.create_evaluator, name, *args, **kwargs
                )
            else:
                raise InvalidOperatorType(op_name)

        return OperatorCreator(database)


def 