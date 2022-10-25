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
    def __init__(self, subgraph: SubGraph) -> None:
        """Graph

        only can be constructed from single subgraph
        but can be divided into multiple subgraphs.
        The subgraph name is maintained within Graph
        """
        super(Graph, self).__init__()
        self.subgraphs = {"0": subgraph}
        self.input_tensors = subgraph.input_tensors
        self.output_tensors = subgraph.output_tensors
        # record the source of tensors if they are the output of subgraphs
        # don't record tensors inside subgraphs
        self._tensors_source = {
            k: "0" for k in self.output_tensors.keys()
        }
