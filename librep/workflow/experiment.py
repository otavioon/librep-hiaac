from collections import defaultdict
from typing import Union, Set, List, Dict, Any
from pathlib import Path

import networkx
import yaml

from librep.base.data import Dataset
from librep.config.type_definitions import ArrayLike, PathLike
from librep.workflow.graph import Operator, GraphBuilder, Executor


class DatasetCreator:
    def __init__(self, root_dir: PathLike):
        self.root_dir = root_dir


class DatasetDatabase:
    def __init__(self, creator: DatasetCreator, database: Dict[str, dict] = None):
        self._creator = creator
        self._database: Dict[str, dict] = database or dict()

    def update_database(self, database: Dict[str, dict], replace: bool = False):
        if replace:
            self._database = database
        else:
            self._database.update(database)

    def load(self, name: str, *args, **kwargs) -> Dataset:
        pass

    @staticmethod
    def from_yaml(creator: DatasetCreator, path: PathLike) -> "DatasetDatabase":
        pass


class ExperimentBuilder:
    def __init__(self, graph_builder: GraphBuilder, dataset_database: DatasetDatabase):
        self.graph_builder = graph_builder
        self.dataset_database = dataset_database
