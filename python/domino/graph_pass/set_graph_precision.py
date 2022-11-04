from typing import Dict, Any, Set, Union, Optional, List

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
        for tname, tensor in new_op.inputs.items():
            if tensor.tensor_idx in self.precision_config:
                tensor.dtype = self.precision_config[tensor.tensor_idx]
        return new_op


class SetUndeterminedOpPrecision(GraphMutator):
    def __init__(
            self,
            graph_inputs_precision: Optional[GeneralDType] = None,
            set_policy: str = "first_set",
            target_ops: Optional[List[Op.OpName]] = None) -> None:
        super(SetUndeterminedOpPrecision, self).__init__()
        self.graph_inputs_precision = graph_inputs_precision
        self.set_policy = set_policy
        assert self.set_policy in [
            "first_set"], f"Unknown policy: {self.set_policy}"
        self.target_ops = set(target_ops) if target_ops is not None else set()

    def mutate_op(self, op: Op.NamedOp, boundary_tensors: Set[Tensor]):
        new_op = super(SetUndeterminedOpPrecision,
                       self).mutate_op(op, boundary_tensors)

        for tname, tensor in new_op.inputs.items():
            if tensor.produce_op is None:
                # graph inputs
                if self.graph_inputs_precision is not None:
                    tensor.dtype = self.graph_inputs_precision

        if op.name not in self.target_ops:
            return new_op

        dtype_for_op = None
        for tname, tensor in new_op.inputs.items():
            if self.set_policy == "first_set":
                if dtype_for_op is None:
                    dtype_for_op = tensor.dtype
            else:
                raise RuntimeError(f"Unknown policy: {self.set_policy}")

        if dtype_for_op is not None:
            for tname, tensor in new_op.outputs.items():
                tensor.dtype = dtype_for_op
            for tname, tensor in new_op.inputs.items():
                tensor.dtype = dtype_for_op

        return new_op


def set_graph_precision(
        graph: Graph,
        precision_config: Dict[Union[int, str], GeneralDType],
        graph_inputs_precision: Optional[GeneralDType] = None,
        set_policy: str = "first_set",
        target_ops: Optional[List[Op.OpName]] = None):
    mutator1 = SetGraphPrecision(precision_config=precision_config)
    mutator2 = SetUndeterminedOpPrecision(
        graph_inputs_precision,
        set_policy=set_policy,
        target_ops=target_ops)
    return mutator2(mutator1(graph))
