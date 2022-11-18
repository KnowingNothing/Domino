from typing import Dict, Any, List, Set, Optional, Union, Tuple
import networkx as nx 
import os 
from itertools import combinations, product
import math
import random
import time 

from domino.utils import ONNXConvertor
from domino.graph_pass import set_graph_precision, GraphPrinter, GraphVisitor
from domino.graph_ir import Op, SubGraph, Graph, Tensor, Attribute
from domino.base import AcceleratorBase, AccTask, AccStream, SoCBase
from domino.accelerator import ConvAccelerator, MeshSoC, NVDLA, GemmTPU, DepthwiseShiDianNao
from domino.program_ir import ConstInt, ConstUInt, ConstFloat, ConstString, ExprList
import matplotlib.pyplot as plt
from domino import global_timer

from base import GraphIRConverter, ComputationGraph, get_graph, MapperBase, visualize

class GrouperBase:
    def group(self, cg: "nx.graph") -> List[List[int]]: 
        raise NotImplementedError()

class PlacerBase:
    def place(self, soc: SoCBase, acc: str, configs: List[Tuple[float, float]],) -> Tuple["Latency", Dict[AccTask, "streamid"], Dict[AccTask, "start_time"]]:
        raise NotImplementedError()

class DPMapper(MapperBase):
    def __init__(self, grouper:GrouperBase, placer: PlacerBase, verbose: bool = False):
        super(DPMapper, self).__init__(verbose)
        self.grouper = grouper 
        self.placer = placer

    
        
    def dp(self, graph: "networkx.graph"):
        key = tuple(sorted([i for i in graph.nodes if not graph.succ[i]]))
        if key not in self.__cache:
            if not len(key):
                self.__cache[key] = (0, [], {}, {})
            else:
                accs = self.soc.get_all_accs()
                best_lat = math.inf
                for frontier in self.grouper.group(graph):
                    frontier_graph = graph.subgraph(frontier)
                    groups = [list(x) for x in  nx.weakly_connected_components(frontier_graph)]
                    remain_graph = graph.subgraph([x for x in graph.nodes if x not in frontier])
                    clock, _, task_placement, _ = self.dp(remain_graph)
                    # todo: consider heterogeneity when doing grouping 
                    candidates = product(*[accs[MapperBase.op2task[self.cg.g.nodes[group[0]]['op'].name]] for group in groups])    
                    for cand in candidates:
                        
                        acc2groups = {}
                        for group, acc in zip(groups, cand):                        
                            if acc not in acc2groups:
                                acc2groups[acc] = []
                            acc2groups[acc].append(group)
                            
                        elapsed_time = 0
                        curr_task_placement = {}
                        curr_task_timing = {}
                        for acc, acc_groups in acc2groups.items():
                            configs = [] # List[Tuple[time, resource]]
                            task_time = {}
                            for gid, group in enumerate(acc_groups):
                                group_graph = frontier_graph.subgraph(group)
                                total_time = 0
                                max_resource_usage = 0
                                for task in nx.topological_sort(group_graph):
                                    task_time[task] = total_time
                                    time, resource_usage = self.soc.eval(
                                        (acc, self.cg.g.nodes[task]['task']), 
                                        [(task_placement[pred][0], self.cg.g.nodes[pred]['task']) for pred in self.cg.g.pred[task] if pred not in group]) 
                                    total_time += time 
                                    max_resource_usage = max(max_resource_usage, resource_usage)
                                configs.append((total_time, max_resource_usage))
                            # placement: List[stream_id], timing: List[float]
                            group_elapsed_time, placement, timing = self.placer.place(self.soc, acc, configs)
                            elapsed_time = max(group_elapsed_time, elapsed_time)
                            for i, group in enumerate(acc_groups):
                                for task in group:
                                    curr_task_placement[task] = (acc, placement[i]) 
                                    curr_task_timing[task] = clock + timing[i] + task_time[task]
                        
                        elapsed_time += clock 
                        if best_lat > elapsed_time:
                            best_lat = elapsed_time
                            best_cand = (elapsed_time, frontier, curr_task_placement, curr_task_timing)
                best_cand[2].update(task_placement)
                self.__cache[key] = best_cand
        
        return self.__cache[key]
    
    def __call__(self, soc: SoCBase):
        self.__cache = {} # Dict[Tuple[Node], Tuple[latency, Dict[task, (Accelerator, Stream)], Dict[task, float]]]]
        self.soc = soc
        self.dp(self.cg.g)
        
        self.g = self.cg.g.copy()
        
        nodes = set(self.g.nodes)
        groups = []
        while len(nodes):
            key = tuple(sorted([nid for nid in nodes if not len(nodes.intersection(self.g.succ[nid]))]))
            assert key in self.__cache 
            _, group, task_placement, task_timing = self.__cache[key]
            nodes.difference_update(group)
            groups = [(group, task_placement, task_timing)] + groups
        
        for group, task_placement, task_timing in groups:
            group.sort(key=lambda t: task_timing[t])
            complete_time = self.commit(soc, group, [task_placement[i] for i in group])
        
        print(f"complete at {complete_time}")
        return complete_time
        
class SimplePlacer(PlacerBase):
    def place(self, soc: SoCBase, acc: "accelerator name", configs: List[Tuple["lat", "resource"]]):
        limits = soc.get_resource_limit(acc)
        stream_limit = limits['num_stream']
        resource_limit = limits['num_pes']
        
        ids = sorted(range(len(configs)), key=lambda x: configs[x][0])
        
        id2stream = {}
        id2time = {}
        
        min_lat = 0
        total_lat = 0
        cur_pe_usage = 0
        i = 0
        stream_id = 0
        for i in range(len(ids)):
            id = ids[i]
            pe_usage = configs[id][1]
            if cur_pe_usage + pe_usage > resource_limit:
                min_lat += configs[ids[i-1]][0]
                cur_pe_usage = 0
                stream_id = 0
            else:
                cur_pe_usage += pe_usage
            id2stream[id] = stream_id 
            id2time[id] = min_lat 
            stream_id = (stream_id + 1) % stream_limit
            if i == len(ids) - 1:
                total_lat = min_lat + configs[id][0]
        return total_lat, id2stream, id2time 

class SimpleGrouper(GrouperBase):
    def group(self, graph):
        return [[i for i in graph.nodes if not graph.succ[i]]]

def main():
    models = ['resnet18', 'resnet50', 'yolo', 'resnet50', 'resnet50']
    graph = get_graph(models)
    print(graph)
    visualize(graph)
    grouper = SimpleGrouper()
    placer = SimplePlacer()
    mapper = DPMapper(grouper, placer)
    cg = ComputationGraph(graph, mapper)
    accs = [[NVDLA("NVDLA(0)", 2), DepthwiseShiDianNao("ShiDianNao(1)"), GemmTPU("GemmTPU")]]
    for acc_list in accs:
        for acc in acc_list:
            print ("acc is", acc.name, acc.num_streams())
    soc = MeshSoC(accs)
    complete_time = cg.map(soc)
    soc.report()
    cg.visualize_packing(soc, complete_time)
    cg.visualize('_'.join(models))
    print("compute lowerbound is ", cg.lower_bound(soc))
    print(f"compute uses {complete_time} ")
    # find (x, y)
if __name__ == "__main__":
    main()
        