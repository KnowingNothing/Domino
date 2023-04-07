from dominoc import passes
from dominoc import ir
import networkx as nx
from functools import reduce


__all__ = ["get_input_tensor_vars",
           "ProdConsumGraph", "get_input_tensor_indices"]


def get_input_tensor_vars(expr):
    return passes.get_input_tensor_vars(expr)


def get_input_tensor_indices(expr, tensor_var):
    return passes.get_input_tensor_indices(expr, tensor_var)


class ProdConsumGraph(object):
    def __init__(self, root_tensor):
        self._make_graph(root_tensor)
        self.node_to_id = {}
        for i, node in enumerate(self.nodes):
            self.node_to_id[node] = i
        self.nx_graph = self._to_networkx_graph()
        self.root_tensor = root_tensor

    def _make_graph(self, root_tensor):
        feed_links = {}
        read_links = {}
        visit = set()
        nodes = []

        def _helper(node):
            if node in visit:
                return
            visit.add(node)
            if node.init is not None:
                for inp in node.init.input_tensors():
                    if node not in read_links:
                        read_links[node] = []
                    read_links[node].append(inp)
                    if inp not in feed_links:
                        feed_links[inp] = []
                    feed_links[inp].append(node)
                    _helper(inp)
                for update in node.updates:
                    for inp in update.input_tensors():
                        if node not in read_links:
                            read_links[node] = []
                        read_links[node].append(inp)
                        if inp not in feed_links:
                            feed_links[inp] = []
                        feed_links[inp].append(node)
                        _helper(inp)
            nodes.append(node)
        _helper(root_tensor)
        self.feed_links = feed_links
        self.read_links = read_links
        self.nodes = nodes

    def _to_networkx_graph(self):
        g = nx.DiGraph()
        for node, reads in self.read_links.items():
            for r in reads:
                g.add_edge(self.node_to_id[node], self.node_to_id[r])
        return g

    def num_nodes(self):
        return len(self.nodes)

    def is_input_tensor(self, tensor):
        return tensor not in self.read_links

    def is_output_tensor(self, tensor):
        return tensor not in self.feed_links or self.feed_links[tensor] == [tensor]

    def dominators(self):
        dom_nodes = nx.immediate_dominators(
            self.nx_graph, self.node_to_id[self.root_tensor])
        ret = {}
        for k, v in dom_nodes.items():
            ret[self.nodes[k]] = self.nodes[v]
        return ret

    def generate_tileflow_workload(self, loops):
        result = "problem:\n"
        result += "  io:\n"
        input_tensors = []
        output_tensors = []
        compute_tensors = []
        for t in self.nodes:
            if self.is_input_tensor(t):
                input_tensors.append(t)
            else:
                compute_tensors.append(t)
            if self.is_output_tensor(t):
                output_tensors.append(t)
        result += f"    ins: {' '.join([str(t.var.id.value) for t in input_tensors])}\n"
        result += f"    outs: {' '.join([str(t.var.id.value) for t in output_tensors])}\n"
        result += f"  dimensions: [{','.join([l.var.id.value for l in loops])}]\n"
        result += f"  instance:\n"
        for l in loops:
            result += f"    {l.var.id.value}: {l.extent.value}\n"
        result += "\n"
        result += "  ops:\n"
        for t in compute_tensors:
            stmt = t.init.as_stmt()
            op_str = passes.generate_tileflow_op(stmt)
            result += op_str
            result += "\n"

        return result
