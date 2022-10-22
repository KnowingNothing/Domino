from typing import List, Dict
from ..base import IRBase
from .op import NamedOp
from .tensor import Tensor

class SubGraph(IRBase):
    def __init__(
        self,
        output_ops: Dict[str, Tensor]
    ) -> None:
        super(SubGraph, self).__init__()
        self.output_ops = output_ops


class Graph(IRBase):
    def __init__(self, subgraphs: List[SubGraph]) -> None:
        super(Graph, self).__init__()
        self.subgraphs = subgraphs
