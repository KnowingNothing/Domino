from typing import List, Dict
from ..base import IRBase
from .op import NamedOp
from .tensor import Tensor


class SubGraph(IRBase):
    def __init__(
        self,
        input_tensors: Dict[str, Tensor],
        output_tensors: Dict[str, Tensor]
    ) -> None:
        super(SubGraph, self).__init__()
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors


class Graph(IRBase):
    def __init__(
            self,
            subgraphs: Dict[str, SubGraph],
            graph_inputs: Dict[str, Tensor],
            graph_outputs: Dict[str, Tensor]) -> None:
        """Graph
        """
        super(Graph, self).__init__()
        self.subgraphs = subgraphs
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs
