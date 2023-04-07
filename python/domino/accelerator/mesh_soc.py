from ..base import SoCBase, MaestroAcceleratorBase
import networkx as nx
from typing import List


class MeshSoC(SoCBase):
    # TODO: OFFCHIP HIGH 120, LOW 60
    def __init__(self, accelerator_matrix: List[List[MaestroAcceleratorBase]], on_chip_bw=32, off_chip_nearest_bw=3.2, name = 'MeshSoC') -> None:
        accelerator_graph = nx.DiGraph()
        self.on_chip_bw = on_chip_bw
        self.off_chip_nearest_bw = off_chip_nearest_bw
        num_rows = len(accelerator_matrix)
        assert num_rows > 0
        num_cols = len(accelerator_matrix[0])
        for row in accelerator_matrix:
            assert len(row) == num_cols

        visited = set()
        for i in range(num_rows):
            for j in range(num_cols):
                acc = accelerator_matrix[i][j]
                acc.topo_id = (i,j)
                assert acc.name not in visited, "Please use unique name for each accelerator instance"
                accelerator_graph.add_node(acc.name, acc=acc)
        for i in range(num_rows):
            for j in range(num_cols):
                for u in range(num_rows):
                    for v in range(num_cols):
                        acc_1 = accelerator_matrix[i][j]
                        acc_2 = accelerator_matrix[u][v]
                        distance = abs(u - i) + abs(v - j)
                        if distance == 0:
                            accelerator_graph.add_edge(
                                acc_1.name, acc_2.name, bandwidth=self.on_chip_bw)
                        else:
                            accelerator_graph.add_edge(
                                acc_1.name, acc_2.name, bandwidth=self.off_chip_nearest_bw / distance)
        super(MeshSoC, self).__init__(accelerator_graph, name=name)
