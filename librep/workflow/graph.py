from collections import defaultdict
from typing import Union, Set, List, Dict, Any
from pathlib import Path

import networkx
import yaml

from librep.config.type_definitions import ArrayLike, PathLike
from librep.base.data import Dataset, SimpleDataset
from librep.base.transform import Transform
from librep.base.estimator import Estimator
from librep.base.evaluators import SupervisedEvaluator
from librep.workflow.objects import ObjectDatabase


class Operator:
    def __init__(self, name: str = ""):
        self.name = name

    def run(self):
        raise NotImplementedError

    def __str__(self) -> str:
        return f"Operator: {self.name}"


class PlaceHolder(Operator):
    def run(self, X):
        return X

    def __str__(self) -> str:
        return f"op=PlaceHolder ({self.name})"


class TransformFitOp(Operator):
    def __init__(
        self, transform: Transform, fit_params: Dict[str, Any] = None, name: str = ""
    ):
        super().__init__(name)
        self.transform = transform
        self.fit_params: Dict[str, Any] = fit_params or dict()

    def run(self, dataset: Dataset) -> Transform:
        X, y = dataset[:][0], dataset[:][1]
        return self.transform.fit(X, y, **self.fit_params)

    def __str__(self) -> str:
        return f"op=TransformFit ({self.name})"


# TODO this should not return a SimpleDataset...
class TransformOp(Operator):
    def run(self, transform: Transform, dataset: Dataset) -> Dataset:
        X = dataset[:][0]
        return SimpleDataset(transform.transform(X), dataset[:][1])

    def __str__(self) -> str:
        return f"op=Transform ({self.name})"


class EstimatorFitOp(Operator):
    def __init__(
        self, estimator: Estimator, fit_params: Dict[str, Any] = None, name: str = ""
    ):
        super().__init__(name)
        self.estimator = estimator
        self.fit_params: Dict[str, Any] = fit_params or dict()

    def run(self, dataset: Dataset) -> Estimator:
        X, y = dataset[:][0], dataset[:][1]
        return self.estimator.fit(X, y, **self.fit_params)

    def __str__(self) -> str:
        return f"op=EstimatorFit ({self.name})"


class EstimatorPredictOp(Operator):
    def run(self, estimator: Estimator, dataset: Dataset) -> ArrayLike:
        X = dataset[:][0]
        return estimator.predict(X)

    def __str__(self) -> str:
        return f"op=EstimatorPredict ({self.name})"


class SupervisedEvaluatorOp(Operator):
    def __init__(self, evaluator: SupervisedEvaluator, name: str = ""):
        super().__init__(name)
        self.evaluator = evaluator

    def run(self, dataset: Dataset, y_pred: ArrayLike) -> Any:
        y_true = dataset[:][1]
        return self.evaluator.evaluate(y_true, y_pred)

    def __str__(self) -> str:
        return f"op=SupervisedEvaluator ({self.name})"


class GraphBuilder:
    def __init__(self, database: ObjectDatabase):
        self.database = database

    def create_node(self, values: dict, name: str = ""):
        if values["type"] == "transform":
            if values["operation"] == "fit":
                obj = self.database.instantiate(values["transform"])
                return TransformFitOp(obj, name=name)
            elif values["operation"] == "transform":
                return TransformOp(name=name)
            else:
                raise ValueError(f"Invalid operation '{values['operation']}'")

        elif values["type"] == "estimator":
            if values["operation"] == "fit":
                obj = self.database.instantiate(values["estimator"])
                return EstimatorFitOp(obj, name=name)
            elif values["operation"] == "predict":
                return EstimatorPredictOp(name=name)
            else:
                raise ValueError(f"Invalid operation '{values['operation']}'")

        elif values["type"] == "evaluator":
            if values["operation"] == "supervised":
                obj = self.database.instantiate(values["evaluator"])
                return SupervisedEvaluatorOp(obj, name=name)
            else:
                raise ValueError(f"Invalid operation '{values['operation']}'")

    def from_dict(
        self, nodes: Dict[str, Any], inputs: List[str] = None, name: str = ""
    ) -> networkx.DiGraph:
        inputs = inputs or []
        graph = networkx.DiGraph(flow=name)

        for i in inputs:
            graph.add_node(i, **{"__op__": PlaceHolder(name=i)})

        for node_name, node_values in nodes.items():
            if "input" in node_values:
                graph.add_edges_from((v, node_name) for v in node_values["input"])
                del node_values["input"]
            node_values["__op__"] = self.create_node(node_values, name=node_name)
            graph.add_node(node_name, **node_values)
        return graph

    def from_yaml(self, path: PathLike) -> Dict[str, networkx.DiGraph]:
        path = Path(path)
        with path.open("r") as f:
            values = yaml.load(f, Loader=yaml.FullLoader)
        results = dict()
        if "flow" in values:
            for flow_name, node_values in values["flow"].items():
                results[flow_name] = self.from_dict(
                    node_values["nodes"], node_values.get("inputs", {}), 
                    name=flow_name
                )
        return results


class Executor:

    def run(self, graph: networkx.DiGraph, placeholders: Dict[str, Any]):
        cache = dict()
        cache.update(placeholders)
        result = None
        print("----- Building done -------\n")

        print(f"Cache: {cache}")
        print()

        topo_sort = list(networkx.topological_sort(graph))
        print(f"Topo sort: {topo_sort}")
        print()

        for node in topo_sort:
            op = graph.nodes[node]["__op__"]
            if isinstance(op, PlaceHolder):
                inputs = [cache[node]]
            else:
                inputs = [cache[u] for (u, v) in graph.in_edges(node)]
            print(f"({node}) executing ({op}): {inputs}")
            result = op.run(*inputs)
            print(f"({node}) result: {result}")
            print()
            cache[node] = result
        return result

# import numpy as np
# from pathlib import Path
# from librep.base.data import SimpleDataset
# from librep.workflow.objects import ObjectCreator, ObjectDatabase
# from librep.workflow.graph import GraphBuilder, Executor
# dataset = SimpleDataset(X=np.arange(16).reshape(2, 8), y=np.array([0, 1]))
# creator = ObjectCreator()
# database = ObjectDatabase.from_yaml(creator=creator, path=Path("objects.yaml"))
# graphs = GraphBuilder(database).from_yaml(Path("workflow.yaml"))
# g = graphs["simple_workflow"]
# h = graphs["simple_workflow_predict"]
# result = Executor().run(g, placeholders={"dataset": dataset})
# result2 = Executor().run(h, placeholders={"dataset": dataset, "trained_model": result})