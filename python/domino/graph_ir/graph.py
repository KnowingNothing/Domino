from typing import List, Dict
from ..base import IRBase
from .op import NamedOp
from .tensor import Tensor
import networkx as nx


class SubGraph(IRBase):
    def __init__(
        self,
        input_tensors: Dict[str, Tensor],
        output_tensors: Dict[str, Tensor]
    ) -> None:
        super(SubGraph, self).__init__()
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors

    @property
    def feed_links(self):
        visit = set()
        ret = {}

        def helper(op: NamedOp):
            nonlocal ret
            nonlocal visit
            if op is None or op in visit:
                return
            visit.add(op)
            for name, inp in op.inputs.items():
                if inp.produce_op is not None and name not in self.input_tensors:
                    if inp.produce_op in ret:
                        ret[inp.produce_op].append(op)
                    else:
                        ret[inp.produce_op] = [op]
                    helper(inp.produce_op)
        for name, t in self.output_tensors.items():
            helper(t.produce_op)
        return ret

    @property
    def read_links(self):
        visit = set()
        ret = {}

        def helper(op: NamedOp):
            nonlocal ret
            nonlocal visit
            if op is None or op in visit:
                return
            visit.add(op)
            for name, inp in op.inputs.items():
                if inp.produce_op is not None and name not in self.input_tensors:
                    if inp.produce_op in ret:
                        ret[op].append(inp.produce_op)
                    else:
                        ret[op] = [inp.produce_op]
                    helper(inp.produce_op)
        for name, t in self.output_tensors.items():
            helper(t.produce_op)
        return ret

    @property
    def ops(self):
        visit = set()
        ret = []

        def helper(op: NamedOp):
            nonlocal ret
            nonlocal visit
            if op is None or op in visit:
                return
            visit.add(op)
            for name, inp in op.inputs.items():
                if inp.produce_op is not None and name not in self.input_tensors:
                    assert inp.produce_op in self.feed_links
                    for consum in self.feed_links[inp.produce_op]:
                        if consum not in visit:
                            continue
                    helper(inp.produce_op)
                    ret.append(op)
        for name, t in self.output_tensors.items():
            helper(t.produce_op)
        return ret

    @property
    def nx_graph(self):
        g = nx.DiGraph()
        node_to_id = {}
        for i, op in enumerate(self.ops):
            node_to_id[op] = i
        for node, reads in self.read_links.items():
            for r in reads:
                g.add_edge(node_to_id[node], node_to_id[r])
        return g


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
