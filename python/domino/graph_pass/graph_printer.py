from typing import Set, Optional, Union, Set, List
from .graph_pass import GraphVisitor
from ..graph_ir import Op, Tensor, SubGraph, Graph


class GraphPrinter(GraphVisitor):
    def __init__(self) -> None:
        super(GraphPrinter, self).__init__()
        self._name_dict = {}
        self.ret = ""
        
    def get_name(self, op: Op.NamedOp):
        pass
    
    def visit_op(self, op: Op.NamedOp, boundary_tensors: Set[Tensor]):
        super(GraphPrinter, self).visit_op(op, boundary_tensors)
        self.ret += f"{op}\n"
        
    def visit_subgraph(self, subgraph: SubGraph, init_state=False):
        self.ret += "Subgraph:\n"
        super(GraphPrinter, self).visit_subgraph(subgraph, init_state)
        
    def visit_graph(self, graph: Graph, specify_subgraphs: Optional[Union[Set[str], List[str]]] = None, init_state=True):
        self.ret += "Graph:\n"
        return super(GraphPrinter, self).visit_graph(graph, specify_subgraphs, init_state)
    
    def __call__(self, graph: Graph, specify_subgraphs: Optional[Union[Set[str], List[str]]] = None, init_state=True) -> str:
        self.ret = ""
        self.visit_graph(graph, specify_subgraphs=specify_subgraphs, init_state=init_state)
        return self.ret
