from typing import List

from librep.workflow.workflow import Workflow, WorkflowNode


class NotAnAcyclicGraph(Exception):
    pass


def topological_sort(workflow: Workflow) -> List[WorkflowNode]:
    # TODO
    raise NotAnAcyclicGraph


class Executor:

    def execute(self, workflow: Workflow, *data):
        raise NotImplementedError


class SimpleExecutor(Executor):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def execute(self, workflow: Workflow, *data):
        sorted_nodes = topological_sort(workflow)
        result = data
        for node in sorted_nodes:
            if self.verbose:
                pass
            result = node(*result)
        return result