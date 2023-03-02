from dominoc import passes
import networkx as nx
from functools import reduce


__all__ = ["get_input_tensor_vars", "ProdConsumGraph"]


def get_input_tensor_vars(expr):
    return passes.get_input_tensor_vars(expr)


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

    def dominators(self):
        dom_nodes = nx.immediate_dominators(
            self.nx_graph, self.node_to_id[self.root_tensor])
        ret = {}
        for k, v in dom_nodes.items():
            ret[self.nodes[k]] = self.nodes[v]
        return ret
