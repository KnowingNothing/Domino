from typing import Dict, Any, Set, Union
from .graph_pass import GraphVisitor, GraphMutator
from ..graph_ir import SubGraph, Graph, Op, Tensor
from ..type_system import GeneralDType


class SetGraphPrecision(GraphMutator):
    def __init__(self, precision_config: Dict[Union[int, str], GeneralDType]) -> None:
        super(SetGraphPrecision, self).__init__()
        self.precision_config = precision_config

    def mutate_op(self, op: Op.NamedOp, boundary_tensors: Set[Tensor]):
        new_op = super(SetGraphPrecision, self).mutate_op(op, boundary_tensors)
        for tname, tensor in new_op.outputs.items():
            if tensor.tensor_idx in self.precision_config:
                tensor.dtype = self.precision_config[tensor.tensor_idx]
        return new_op

    def __call__(self, graph: Graph) -> Any:
        return self.mutate_graph(graph)


def set_graph_precision(graph: Graph, precision_config: Dict[Union[int, str], GeneralDType]):
    mutator = SetGraphPrecision(precision_config=precision_config)
    return mutator(graph)
