from typing import Dict, Any, List, Set, Optional, Union, Tuple
import networkx as nx 
import os 
from itertools import combinations, product
import math
import random
import time 
import numpy as np

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


class Individual:
    def __init__(self):
        self.latency = math.inf
        self.topo_order = list()
        self.layers: List[Tuple[float, List[int]]] = list()
    
    def __str__(self):
        return f'lat={self.latency},ave_occupancy={np.mean([x[0]for x in self.layers])}'
    
    def __repr__(self):
        return f'lat={self.latency},ave_occupancy={np.mean([x[0]for x in self.layers])}'
    
class Population:
    num_generations = 10
    size = 4
    direct_heir_rate = 0.2
    mutate_rate = 0.1
    
    def __init__(self):
        self.population:List[Individual] = list()
        
    def select(self, n = 1, best = False) -> Union[List[Individual], Individual]:
        if best:
            best_lat = math.inf 
            for idv in self.population:
                if idv.latency < best_lat: 
                    best_lat = idv.latency 
                    best_idv = idv 
            return best_idv 

        scores = - np.array([x.latency for x in self.population])
        normalized = (scores - np.mean(scores)) / np.std(scores)
        exp = np.exp(normalized)
        probs = exp / np.sum(exp)
        return np.random.choice(self.population, n, p=probs)            
            
    def add(self, x: Individual):
        self.population.append(x)

    def show(self):
        for i, idv in enumerate(self.population):
            print(f'{i}: {idv}')
class EvolutionMapper(MapperBase):
    def __init__(self, topo_sort_alg:str, placer: PlacerBase, verbose: bool = False):
        super(EvolutionMapper, self).__init__(verbose)
        self.topo_sort_alg = topo_sort_alg
        self.placer = placer
        self.max_depth = 20
    # dp[:idx] = max_{j}{dp[:idx-j] + self.placer(dp[idx-j:idx])}  
    def dp(self):
        self.__cache[0] = (0, 0, {}, {}, {})
        accs = self.soc.get_all_accs()
        for idx in range(1, len(self.cg.g.nodes)+1):
            best_lat = math.inf
            for j in range(1, min(idx, self.max_depth)+1):
                frontier_graph = self.cg.g.subgraph(self.topo_order[idx-j:idx]) 
                
                groups = [list(x) for x in  nx.weakly_connected_components(frontier_graph)]
                clock, _, task_placement, _, _ = self.__cache[idx - j]
                # todo: consider heterogeneity when doing grouping 
                candidates = product(*[accs[MapperBase.op2task[self.cg.g.nodes[group[0]]['op'].name]] for group in groups])
                n_avail_cand = 0
                for cand in candidates:
                    # special judgement for heterogeneity mapping
                    valid = True
                    for group, acc in zip(groups, cand):
                        for task in group:
                            if acc not in accs[MapperBase.op2task[self.cg.g.nodes[task]['op'].name]]:
                                valid = False 
                    if not valid: continue 
                    n_avail_cand += 1
                    
                    
                    acc2groups = {}
                    for group, acc in zip(groups, cand):                        
                        if acc not in acc2groups:
                            acc2groups[acc] = []
                        acc2groups[acc].append(group)
                        
                    elapsed_time = 0
                    curr_task_placement = {}
                    curr_task_timing = {}
                    curr_hardware_occupancy = {}
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
                        group_elapsed_time, placement, timing, attrs = self.placer.place(self.soc, acc, configs)
                        elapsed_time = max(group_elapsed_time, elapsed_time)
                        curr_hardware_occupancy[acc] = attrs['occupancy']
                        for i, group in enumerate(acc_groups):
                            for task in group:
                                curr_task_placement[task] = (acc, placement[i]) 
                                curr_task_timing[task] = clock + timing[i] + task_time[task]
                    if best_lat > elapsed_time + clock:
                        best_lat = elapsed_time + clock
                        best_cand = (elapsed_time + clock, j, curr_task_placement, curr_task_timing, curr_hardware_occupancy)

            assert best_lat < math.inf
                        
            best_cand[2].update(self.__cache[idx - best_cand[1]][2])
            self.__cache[idx] = best_cand
                
        return self.__cache[len(self.cg.g.nodes)]
   
    def bfs(self, graph):
        L = []
        num_preds = {i:len(graph.pred[i]) for i in graph.nodes}
        S = [i for i, n_pred in num_preds.items() if n_pred == 0]
        while len(S):
            i = S[0]
            L.append(i)
            S = S[1:]
            succs = list(graph.succ[i])
            random.shuffle(succs)
            for j in succs:
                num_preds[j] -= 1 
                if num_preds[j] == 0:
                    S.append(j)
        return L 

    def bfs_reversed(self, graph):
        L = []
        num_succs = {i:len(graph.succ[i]) for i in graph.nodes}
        S = [i for i, n_succ in num_succs.items() if n_succ == 0]
        while len(S):
            i = S[0]
            L.append(i)
            S = S[1:]
            preds = list(graph.pred[i])
            random.shuffle(preds)
            for j in preds:
                num_succs[j] -= 1
                if num_succs[j] == 0:
                    S.append(j)
        L.reverse()
        
        return L 

    def dfs(self, graph):
        L = []
        visited = set()
        def visit(i):
            if i in visited:
                return
         
            visited.add(i)
            preds = list(graph.pred[i])
            random.shuffle(preds)
            for pred in preds:
                visit(pred)
            nonlocal L 
            L.append(i) 
                
        for i in graph.nodes:
            visit(i)
        return L 

    def dfs_reversed(self, graph):
        L = []
        visited = set()
        def visit(i):
            if i in visited: return 
            visited.add(i)
            succs = list(graph.succ[i])
            random.shuffle(succs)
            for succ in succs:
                visit(succ)
            nonlocal L 
            L.append(i)
        for i in graph.nodes: visit(i)
        L.reverse()
        return L 

    def topo_sort(self, graph):
        return [self.dfs, self.bfs, self.dfs_reversed, self.bfs_reversed][random.randint(0,3)](graph)

    def __call__(self, soc: SoCBase):
        self.soc = soc
        
        self.closure_graph = nx.transitive_closure(self.cg.g)
        idv = self.evolve_search()
        
        self.topo_order = idv.topo_order
        
        print (self.topo_order)
        
        self.__cache = {} # Dict[Tuple[Node], Tuple[latency, Dict[task, (Accelerator, Stream)], Dict[task, float]]]]
        
        self.dp()
        
        groups = []
        i = len(self.cg.g.nodes)
        while i != 0:
            _, j, task_placement, task_timing, _ = self.__cache[i]
            groups = [((i-j, i), task_placement, task_timing)] + groups
            i -= j 
                    
        self.g = self.cg.g.copy()
        for r, task_placement, task_timing in groups:
            group = self.topo_order[r[0]:r[1]]
            group.sort(key=lambda t: task_timing[t])
            print(group)
            complete_time = self.commit(soc, group, [task_placement[i] for i in group])
        
        print(f"complete at {complete_time}, estimated {self.__cache[len(self.cg.g.nodes)][0]}")
        return complete_time
    
    def init_population(self):
        pop = Population()
        for _ in range(Population.size):
            pop.add(self.eval(self.topo_sort(self.cg.g)))
        return pop 
    
    def eval(self, topo_order:List[int]):
        ret = Individual()
        ret.topo_order = topo_order 
        self.topo_order = topo_order 
        self.__cache = {}
        ret.latency = self.dp()[0]
        ret.layers = []

        i = len(self.cg.g.nodes)
        while i != 0:
            _, j, _, _, acc_occupancy = self.__cache[i]
            ret.layers.append((np.mean(list(acc_occupancy.values())), ret.topo_order[i-j:i]))
            i -= j
        
        return ret
        
    def gen_idv_with_fixed(self, fixed: List[List[int]]):
        graph = self.cg.g.copy()
        old2new = {i:i for i in graph.nodes}
        new_edges = []
        for i, layer in enumerate(fixed):
            for node in layer:
                old2new[node] = -(i+1)
                new_edges += list(graph.in_edges(node)) + list(graph.out_edges(node)) 

        graph.remove_nodes_from(sum(fixed, []))
        graph.add_nodes_from([-(i+1) for i in range(len(fixed))])    
        graph.add_edges_from(set([(old2new[u], old2new[v]) for u,v in new_edges if old2new[u] != old2new[v]]))

        fake_topo_order = self.topo_sort(graph)    
        topo_order = sum([fixed[-i-1] if i < 0 else [i] for i in fake_topo_order], [])
        
        return self.eval(topo_order)
    
    def crossover(self, x: Individual, y: Individual):
        parents = [x,y]
        layer_ids = []
        for i, parent in enumerate(parents):
            for j in range(len(parent.layers)):
                layer_ids.append((i,j))
        layer_ids = sorted(layer_ids, key=lambda x: parents[x[0]].layers[x[1]][0], reverse = True)
        fixed = []
        invalid = set()
        
        def compatible(group1, group2):
            forward = False 
            backward = False
            for i in group1:
                for j in group2:
                    if i == j:
                        return False 
                    if self.closure_graph.has_edge(i,j):
                        forward = True 
                        if backward: return False 
                    if self.closure_graph.has_edge(j,i):
                        backward = True 
                        if forward: return False 
            return True 
        
        for i, j in layer_ids:
            if (i,j) in invalid: continue 
            fixed.append(parents[i].layers[j][1])
            ii = 1 - i 
            for jj in range(len(parents[ii].layers)):
                if (ii, jj) in invalid: continue
                if not compatible(parents[i].layers[j][1], parents[ii].layers[jj][1]):
                    invalid.add((ii, jj))
        return self.gen_idv_with_fixed(fixed)
        
    def mutate(self, x: Individual):
        print(f'x is {x}', type(x))
        fixed = []
        for layer in x.layers:
            if random.random() > Population.mutate_rate: 
                fixed.append(layer[1])
        return self.gen_idv_with_fixed(fixed)
    
    def evolve_search(self):
        pop = self.init_population()
        
        for generation in range(Population.num_generations):
            pop.show()
            print (f"generation {generation}, score {pop.select(best=True)}")
            new_pop = Population()
            for i in range(Population.size):
                if random.random() < Population.direct_heir_rate:
                    # direct pass the chronosome down
                    print(f"{generation}, {i}: direct inherit") 
                    offspring = pop.select()[0]
                else:
                    print(f'{generation}, {i}: crossover')
                    x, y = pop.select(2)
                    offspring = self.crossover(x,y)
                print (f'offspring:{offspring}')
                offspring = self.mutate(offspring)
                new_pop.add(offspring)
            pop = new_pop 
            
        return pop.select(best = True)
    
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
        attrs = {'n_increase': 0}
        for i in range(len(ids)):
            id = ids[i]
            pe_usage = configs[id][1]
            if cur_pe_usage + pe_usage > resource_limit:
                min_lat += configs[ids[i-1]][0]
                cur_pe_usage = 0
                stream_id = 0
                attrs['n_increase'] += 1
            else:
                cur_pe_usage += pe_usage
            id2stream[id] = stream_id 
            id2time[id] = min_lat 
            stream_id = (stream_id + 1) % stream_limit
            if i == len(ids) - 1:
                total_lat = min_lat + configs[id][0]
        attrs['occupancy'] = sum([x*y for x, y in configs])/ (resource_limit * total_lat)
        return total_lat, id2stream, id2time, attrs 

def main():
    models = ['resnet18', 'resnet50', 'yolo', 'resnet50', 'resnet50']
    graph = get_graph(models)
    print(graph)
    visualize(graph)
    placer = SimplePlacer()
    mapper = EvolutionMapper('bfs', placer)
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
    print("compute uses ", complete_time)

    # find (x, y)

if __name__ == "__main__":
    random.seed(1)
    os.system("mkdir -p .cache")
    AcceleratorBase.load_cache()
    main()
    AcceleratorBase.store_cache()
