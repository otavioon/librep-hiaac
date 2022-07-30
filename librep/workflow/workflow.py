from typing import Iterable, Dict, Optional, Union, Callable
from uuid import uuid4

from librep.base.estimator import Estimator
from librep.base.transform import Transform
from librep.config.type_definitions import KeyType


class WorkflowNode:

    def __init__(self, name: str = "", description: str = ""):
        self.node_id = str(uuid4())
        self.name = name
        self.description = description

    def __str__(self) -> str:
        return f"node {self.node_id} ({self.name})"

    def __call__(self, *args):
        raise NotImplementedError


class Workflow:

    def __init__(self, name: str = "", description: str = ""):
        self.node_id = str(uuid4())
        self.name = name
        self.description = description
        self.nodes = dict()
        self.edges = dict()
        self.start_node = None

    @staticmethod
    def from_nodes_edges(nodes: Dict[KeyType, WorkflowNode],
                         edges: Dict[KeyType, KeyType],
                         start_node: KeyType) -> "Workflow":
        workflow = Workflow()
        for key, val in nodes.items():
            workflow.add_node(key, val)
        for from_node, to_node in edges.items():
            workflow.add_edge(from_node, to_node)
        workflow.set_start_node(start_node)
        return workflow

    def add_edge(self, from_node: KeyType, to_node: KeyType):
        assert from_node in self.nodes, f"Invalid node (from) {from_node}"
        assert to_node in self.nodes, f"Invalid node (to) {to_node}"
        self.edges[from_node] = to_node

    def add_node(self, key: KeyType, value: WorkflowNode):
        self.nodes[key] = value

    def set_start_node(self, key: KeyType):
        assert key in self.nodes, f"Invalid start node: {key}"
        self.start_node = key


####### Wrappers #########


class CallableNode(WorkflowNode):

    def __init__(self, fn: callable, name: str = "", description: str = ""):
        super().__init__(name, description)
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class EstimatorNode:

    def __init__(self,
                 estimator: Estimator,
                 action: str = "fit_predict",
                 name: str = "",
                 description: str = "",
                 **estimator_params):
        super().__init__(name, description)
        self.estimator = estimator
        self.action = action
        self.estimator_params = estimator_params

    def __call__(self, *args):
        if self.action == "fit":
            return self.estimator.fit(*args, **self.estimator_params)
        if self.action == "predict":
            return self.estimator.predict(*args)
        if self.action == "fit_predict":
            return self.estimator.fit_predict(*args, **self.estimator_params)
        else:
            raise ValueError(f"Invalid estimator action: '{self.action}'")


class TransformNode:

    def __init__(self,
                 transform: Transform,
                 action: str = "fit_transform",
                 name: str = "",
                 description: str = "",
                 **fit_params):
        super().__init__(name, description)
        self.transform = transform
        self.action = action
        self.fit_params = fit_params

    def __call__(self, *args):
        if self.action == "fit":
            return self.estimator.fit(*args, **self.fit_params)
        if self.action == "transform":
            return self.estimator.transform(*args)
        if self.action == "fit_transform":
            return self.estimator.fit_transform(*args, **self.fit_params)
        else:
            raise ValueError(f"Invalid transform action: '{self.action}'")


NodeType = Union[WorkflowNode, Callable]

node_wrappers = {Estimator: EstimatorNode, Transform: TransformNode}


def make_workflow(actions: Union[Iterable[NodeType], Dict[KeyType, NodeType]],
                  graph: Optional[Dict[KeyType, NodeType]] = None,
                  start_node: Optional[KeyType] = None) -> Workflow:

    # Simple Type checking...
    if isinstance(actions, dict):
        assert graph is not None, "graph variable must be defined for graph type"
        assert start_node is not None, "start_node variable must be set for graph type"
    elif not isinstance(actions, Iterable):
        raise ValueError(f"Invalid type for actions variable: {type(actions)}")

    if not isinstance(actions, dict):
        actions = {i: action for i, action in enumerate(actions)}
        graph = {i: i + 1 for i in enumerate(actions[:-1])}
        start_node = 0

    # Convert all actions to WorkflowNode type...
    _actions = dict()
    for i, (key, value) in enumerate(_actions):
        if type(value) in node_wrappers:
            node = node_wrappers[type(value)](value)
        elif not isinstance(value, WorkflowNode):
            node = CallableNode(value)
        else:
            node = value
        # TODO check if type is an workflow (a recursive) -- Unpack it to a single graph
        _actions[key] = node

    #TODO
    return Workflow()