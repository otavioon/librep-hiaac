import importlib
import inspect
import pkgutil
import yaml

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


class ObjectCreator:

    def __init__(self,
                 transform_lib: str = "librep.transforms",
                 estimator_lib: str = "librep.estimators",
                 evaluator_lib: str = "librep.evaluators"):
        self.transform_classes = {
            "FFT":
                librep.transforms.fft.FFT,
            "AutoCorrelation":
                librep.transforms.autocorrelation.AutoCorrelation,
            "SimpleResampler":
                librep.transforms.resampler.SimpleResampler,
            "Spectrogram":
                librep.transforms.spectrogram.Spectrogram,
            "StatsTransform":
                librep.transforms.stats.StatsTransform,
            "TSNE":
                librep.transforms.tsne.TSNE,
            "UMAP":
                librep.transforms.umap.UMAP,
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


class ObjectDatabase:

    def __init__(self,
                 creator: ObjectCreator,
                 database: Dict[str, dict] = None):
        self._creator = creator
        self._database: Dict[str, dict] = database or dict()

    def update_database(self, database: Dict[str, dict], replace: bool = False):
        if replace:
            self._database = database
        else:
            self._database.update(database)

    def instantiate(self, name: str, *args, **kwargs):
        values = self._database[name]
        # build args
        new_args = args or values.get("args", tuple())
        # build the kwargs
        new_kwargs = values.get("kwargs", {})
        new_kwargs.update(kwargs)

        if "transform" in values:
            name = values["transform"]
            return self._creator.create_transform(name, *new_args, **new_kwargs)
        elif "estimator" in values:
            name = values["estimator"]
            return self._creator.create_estimator(name, *new_args, **new_kwargs)
        elif "evaluator" in values:
            name = values["evaluator"]
            return self._creator.create_evaluator(name, *new_args, **new_kwargs)
        else:
            raise ValueError("Invalid value")

    @staticmethod
    def from_yaml(creator: ObjectCreator, path: PathLike) -> "ObjectDatabase":
        with Path(path).open("r") as f:
            objs = yaml.load(f, Loader=yaml.FullLoader)
        return ObjectDatabase(creator, database=objs["objects"])

