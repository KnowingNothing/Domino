from typing import Dict, Any, List, Set, Optional, Union, Tuple
import networkx as nx 
import os 
from itertools import combinations, product
import math
import random
import time 
import numpy as np
from functools import reduce
import pickle as pkl

from domino.utils import ONNXConvertor
from domino.graph_pass import set_graph_precision, GraphPrinter, GraphVisitor
from domino.graph_ir import Op, SubGraph, Graph, Tensor, Attribute
from domino.base import MaestroAcceleratorBase, AccTask, AccStream, SoCBase
from domino.accelerator import MaestroConvAccelerator, MeshSoC, MaestroNVDLA, MaestroGemmTPU, MaestroDepthwiseShiDianNao, MaestroConvShiDianNao
from domino.program_ir import ConstInt, ConstUInt, ConstFloat, ConstString, ExprList
import matplotlib.pyplot as plt
from domino import global_timer

from base import GraphIRConverter, ComputationGraph, get_graph, MapperBase, visualize, visualize_basic, visualize_subgraph, ParallelMachineScheduler, visualize_serialize

class GrouperBase:
    def group(self, cg: "nx.graph") -> List[List[int]]: 
        raise NotImplementedError()

class PlacerBase:
    def place(self, soc: SoCBase, acc: str, configs: List[Tuple[float, float]],) -> Tuple["Latency", Dict[AccTask, "streamid"], Dict[AccTask, "start_time"], Dict[str, Any]]:
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
    num_generations = 20
    size = 8
    mutate_rate = 0.3
    layer_mutate_rate = 0.8
    
    def __init__(self):
        self.population:List[Individual] = list()
        
    def natrual_choice(self):
        self.population.sort(key=lambda idv: idv.latency)
        self.population = self.population[:Population.size]
        
    def get_best_lat(self):
        return min(idv.latency for idv in self.population)    
    
    def select(self, n = 1, best = False) -> Union[List[Individual], Individual]:
        if best:
            best_lat = math.inf 
            for idv in self.population:
                if idv.latency < best_lat: 
                    best_lat = idv.latency 
                    best_idv = idv 
            return best_idv 
        
        scores = - np.array([x.latency for x in self.population])
        normalized = (scores - np.mean(scores)) / (np.std(scores) + 1)
        exp = np.exp(normalized)
        probs = exp / np.sum(exp)
        return np.random.choice(self.population, n, p=probs)            
            
    def add(self, x: Individual):
        self.population.append(x)

    def show(self):
        for i, idv in enumerate(self.population):
            print(f'{i}: {idv}')
            
            
class EvolutionMapper(MapperBase):
    def __init__(self, placer: Union[PlacerBase, None] = None, scheduler: Union[ParallelMachineScheduler, None] = None, file_path:Union[str, None] = None, verbose: bool = False, cached = False):
        super(EvolutionMapper, self).__init__(verbose)
        self.file_path = file_path
        self.placer = placer
        self.cached = cached
        self.scheduler = scheduler
        self.max_depth = 20
        self.max_mapping_candidate = 100
        
    # dp[:idx] = max_{j}{dp[:idx-j] + self.placer(dp[idx-j:idx])}  
    def dp(self):
        self.__cache.clear()
        self.__cache[0] = (0, 0, {}, {}, {})
        accs = self.soc.get_all_accs()
        compute_times = self.get_compute_time(accs)
        resource_usages = self.get_resource_usage(accs)
        reversed_topo_order = {x:i for i,x in enumerate(self.topo_order)}
        acc2task_kinds = self.soc.get_acc2task_kinds()
        
        for idx in range(1, len(self.cg.g.nodes)+1):
            best_lat = math.inf
            for j in range(1, min(idx, self.max_depth)+1):
                frontier_graph = self.cg.g.subgraph(self.topo_order[idx-j:idx]) 
                
                groups = [sorted(list(x), key=lambda y: reversed_topo_order[y]) for x in  nx.weakly_connected_components(frontier_graph)]
                
                # if exist one group that no device support all ops, continue 
                invalid = False
                for group in groups:
                    has_avail_acc = False
                    for task_kinds in acc2task_kinds.values():
                        valid = True 
                        for nid in group:
                            task_kind = self.cg.g.nodes[nid]['task'].task_kind
                            if task_kind not in task_kinds: 
                                valid = False 
                                break 
                        if valid: 
                            has_avail_acc = True  
                            break
                    if not has_avail_acc: 
                        invalid = True
                        break 
                if invalid: continue
                
                clock, _, task_placement, _, _ = self.__cache[idx - j]
                
                group_times = [] # List[Dict[Accelerator, List[time]]]
                group_resource_usage = [] 
                for group in groups:
                    acc_times = {}
                    acc_resource_usage = {}
                    for acc in accs[MapperBase.op2task[self.cg.g.nodes[group[0]]['op'].name]]:
                        start_times = [0]
                        for nid in group:
                            compute_time = compute_times[nid][acc]
                            comm_time = self.soc.eval_communication(
                                    (acc, self.cg.g.nodes[nid]['task']), 
                                    [(task_placement[pred][0] if pred not in group else acc, self.cg.g.nodes[pred]['task']) for pred in self.cg.g.pred[nid]]) 
                            start_times.append(start_times[-1] + (compute_time + comm_time))
                        acc_times[acc] = start_times
                        acc_resource_usage[acc] = max(resource_usages[nid][acc] for nid in group)
                    group_times.append(acc_times)
                    group_resource_usage.append(acc_resource_usage)
                
                if self.scheduler is not None:
                    machines = self.soc.get_machines()
                    resource_constraint = self.soc.get_all_resource_limit()
                    resource_requirement = group_resource_usage
                    timing = []
                    for x in group_times:
                        timing.append({acc: x[acc][-1] for acc in x})
                    if not len(timing):
                        print (f'ERROR, idx{idx}, j{j}, group_times{group_times}, groups{groups}')
                    elapsed_time, placement, scheduling, hardware_occupancy = self.scheduler.schedule(2, machines, resource_constraint, resource_requirement, timing)
                    if best_lat > elapsed_time + clock:
                        best_lat = elapsed_time + clock 
                        curr_task_placement = {}
                        curr_task_timing = {}
                        for i, group in enumerate(groups):
                            machine, resource = placement[i]
                            start_time = clock + scheduling[i] 
                            for ii, nid in enumerate(group):
                                curr_task_placement[nid] = (machine, resource[0])
                                assert resource[0] < self.soc.accelerator_graph.nodes[machine]['acc'].num_streams()
                                curr_task_timing[nid] = start_time 
                                start_time += group_times[i][machine][ii]
                        best_cand = (best_lat, j, curr_task_placement, curr_task_timing, {x: hardware_occupancy[x][0] for x in hardware_occupancy})    
                # todo: consider heterogeneity when doing grouping 
                elif self.placer is not None:
                    candidates = list(product(*[accs[MapperBase.op2task[self.cg.g.nodes[group[0]]['op'].name]] for group in groups]))
                    
                    for cand in random.sample(candidates, min(len(candidates), self.max_mapping_candidate)):     
                        acc2groups = {}
                        for gid, acc in enumerate(cand):                        
                            if acc not in acc2groups:
                                acc2groups[acc] = []
                            acc2groups[acc].append(gid)
                            
                        elapsed_time = 0
                        curr_task_placement = {}
                        curr_task_timing = {}
                        curr_hardware_occupancy = {}

                        for acc, acc_groups in acc2groups.items():
                            group_elapsed_time, placement, timing, attrs = self.placer.place(
                                self.soc, acc, [(group_times[gid][acc][-1], group_resource_usage[gid][acc][1]) for gid in acc_groups])
                            
                            elapsed_time = max(group_elapsed_time, elapsed_time)
                            
                            if elapsed_time + clock > best_lat: break 
                            
                            curr_hardware_occupancy[acc] = attrs['occupancy']
                            for i, gid in enumerate(acc_groups):
                                for ii, nid in enumerate(groups[gid]):
                                    curr_task_placement[nid] = (acc, placement[i]) 
                                    curr_task_timing[nid] = clock + timing[i] + group_times[gid][acc][ii]
                                    
                        if best_lat > elapsed_time + clock:
                            best_lat = elapsed_time + clock
                            best_cand = (best_lat, j, curr_task_placement, curr_task_timing, curr_hardware_occupancy)
                else:
                    raise RuntimeError('at least one of scheudler and placer is not none')
            assert best_lat < math.inf
            best_cand[2].update(self.__cache[idx - best_cand[1]][2])
            self.__cache[idx] = best_cand
                
        return self.__cache[len(self.cg.g.nodes)]
   
    def topo_check(self, topo_order):
        invalid = False
        visited = set()
        for nid in topo_order:
            for pred in self.cg.g.pred[nid]:
                if not pred in visited: 
                    print (f"{pred}->{nid} violation")
                    invalid = True
            visited.add(nid)
        return not invalid 

    def check_idv(self, idv):
        if not self.topo_check(idv.topo_order):
            return False 
        nodes = set()
        for layer in idv.layers:
            for i in layer[1]:
                nodes.add(i)
        if not len(nodes) == len(self.cg.g.nodes):
            print("layer error")
            return False
        return True
   
    def fuzz_test(self, soc):
        self.soc = soc
        self.closure_graph = nx.transitive_closure(self.cg.g)
        
        
        for name, alg in [('bfs_dfs', self.bfs_dfs), ('bfs_dfs', self.bfs_dfs_reversed), ('bfs', self.bfs), ('bfs_reversed', self.bfs_reversed), ('dfs', self.dfs), ('dfs_reversed', self.dfs_reversed)]:
            for i in range(20):
                topo_order = alg(self.cg.g)
                assert self.topo_check(topo_order)
        idvs = []
        n_test = 100
        for i in range(n_test):        
            idvs.append(self.eval(self.topo_sort(self.cg.g)))
            assert self.check_idv(idvs[-1])
        for i in range(n_test):
            idvs[i] = self.mutate(idvs[i])
            assert self.check_idv(idvs[i])
        for i in range(n_test):
            x = random.randint(0, n_test-1)
            y = random.randint(0, n_test-1)
            idvs[i] = self.crossover(idvs[x],idvs[y])
            assert self.check_idv(idvs[i])
        print('test passed!')
    
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

    def bfs_dfs(self, graph):
        L = []
        num_preds = {i:len(graph.pred[i]) for i in graph.nodes}
        S = [i for i, n_pred in num_preds.items() if n_pred == 0]
        while len(S):
            i = S[0]
            L.append(i)
            S = S[1:]
            succs = list(graph.succ[i])
            for j in succs:
                num_preds[j] -= 1 
                if num_preds[j] == 0:
                    if random.random() < 0.5:
                        S.append(j)
                    else: S = [j] + S
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
    
    def bfs_dfs_reversed(self, graph):
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
                    if random.random() < 0.5:
                        S.append(j)
                    else: S = [j] + S
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
    
    def exist_cycle(self, graph):
        visited = set()
        path = set()
        def visit(i):
            if i in visited:
                return
         
            visited.add(i)
            preds = list(graph.pred[i])
            for pred in preds:
                if pred in path: return True 
                path.add(pred)
                visit(pred)
                path.remove(pred)
                
        for i in graph.nodes:
            path.add(i)
            visit(i)
            path.remove(i)
        return False

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

    def random_sort(self, graph):
        L = []
        pred_cnt = {x:len(graph.pred[x]) for x in graph.nodes}
        
        frontiers = [x for x,cnt in pred_cnt.items() if cnt == 0]
        
        while len(frontiers):
            i = random.randint(0,len(frontiers)-1)
            nid = frontiers[i]
            L.append(frontiers[i])
            frontiers = frontiers[:i] + frontiers[i+1:]
            for succ in graph.succ[nid]:
                pred_cnt[succ] -= 1
                if pred_cnt[succ] == 0:
                    frontiers.append(succ)
        return L 

    def topo_sort(self, graph):
        return [self.bfs, self.bfs_reversed, self.bfs_dfs, self.bfs_dfs_reversed][random.randint(0,3)](graph)

    def __call__(self, soc: SoCBase):
        self.soc = soc
        
        if self.cached and os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                self.topo_order = pkl.load(f)
        else:
            self.closure_graph = nx.transitive_closure(self.cg.g)
            idv = self.evolve_search()
            self.topo_order = idv.topo_order
                
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
            complete_time = self.commit(soc, group, [task_placement[i] for i in group])
            # complete_time = self.commit_rr(soc, group)

        if self.file_path is not None:
            with open(self.file_path, 'wb') as f:
                pkl.dump(self.topo_order, f)
        
        print(f"complete at {complete_time}, estimated {self.__cache[len(self.cg.g.nodes)][0]}")
        return complete_time
    
    def init_population(self):
        pop = Population()
        for _ in range(Population.size):
            pop.add(self.eval(self.dfs(self.cg.g)))
            #pop.add(self.eval(self.bfs_dfs(self.cg.g)))
        return pop 
    
    def eval(self, topo_order:List[int]):
        ret = Individual()
        ret.topo_order = topo_order 
        self.topo_order = topo_order 
        self.__cache = {}
        ret.latency = self.dp()[0]
        assert ret.latency < math.inf
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
        
        if not self.topo_check(topo_order):
            visualize_basic(graph, 'error')
            visualize_subgraph(self.cg.g, fixed, 'error_raw')
            for i, l in enumerate(fixed):
                print (-(i+1), ":", l)
            for nid in self.cg.g.nodes:
                print(nid, ':', self.cg.g.pred[nid])
            assert False
            
        return self.eval(topo_order)
    
    def crossover(self, x: Individual, y: Individual):
        parents = [x,y]
        layer_ids = []
        for i, parent in enumerate(parents):
            for j in range(len(parent.layers)):
                layer_ids.append(j*2+i)
        layer_ids = sorted(layer_ids, key=lambda x: parents[x%2].layers[x//2][0], reverse = True)
        
        def has_edge(x, y):
            i, j = x%2, x//2
            ii, jj = y%2, y//2
            for v in parents[i].layers[j][1]:
                for vv in parents[ii].layers[jj][1]:
                    if v == vv or self.closure_graph.has_edge(v, vv): 
                        return True
            return False 
            
        compatible_graph = nx.DiGraph()
        for id in layer_ids:
            tos  = set((iidd for iidd in compatible_graph.nodes if has_edge(id, iidd)))
            tos = tos.union(nid for to in tos for nid in compatible_graph.succ[to])
            froms = set(iidd for iidd in compatible_graph.nodes if has_edge(iidd, id))
            froms = froms.union(nid for from_id in froms for nid in compatible_graph.pred[from_id])
            
            if tos & froms:
                continue 
            
            compatible_graph.add_node(id)
            tos.add(id)
            froms.add(id)
            compatible_graph.add_edges_from([(u, v) for u in froms for v in tos if u != v])            
                        
        fixed = [parents[x%2].layers[x//2][1] for x in compatible_graph.nodes]                            


        return self.gen_idv_with_fixed(fixed)
        
    def mutate(self, x: Individual):
        fixed = []
        ids = sorted(range(len(x.layers)), key = lambda i: x.layers[i][0], reverse=True)
        n_fixed = int(len(x.layers) * (1 - Population.layer_mutate_rate))
        fixed = [x.layers[i][1] for i in ids[:n_fixed]]
        # for layer in x.layers:
        #     if random.random() > Population.layer_mutate_rate: 
        #         fixed.append(layer[1])
        return self.gen_idv_with_fixed(fixed)
    
    def export(self, key):
        with open(f"evolve_search/{key}.pkl", 'wb') as f:
            pkl.dump(self.best_lats, f)
    
    def evolve_search(self):
        pop = self.init_population()
        best_idv = pop.select(best=True)
        self.best_lats = []
        for generation in range(Population.num_generations):
            self.best_lats.append(pop.get_best_lat())
            pop.show()
            new_pop = Population()
            print (f"generation {generation}, score {pop.select(best=True)}")
            for i in range(Population.size):
                if random.random() < Population.mutate_rate:
                    # direct pass the chronosome down
                    print(f"{generation}, {i}: mutate") 
                    offspring = self.mutate(pop.select()[0])
                else:
                    print(f'{generation}, {i}: crossover')
                    x, y = pop.select(2)
                    offspring = self.crossover(x,y)
                print (f'offspring:{offspring}')
                new_pop.add(offspring)
                if best_idv.latency > offspring.latency:
                    best_idv = offspring
            pop = new_pop
        self.best_lats.append(pop.get_best_lat())
        return best_idv
    
class SimplePlacer(PlacerBase):
    # stream_limit limits the maximum stream used, 20 is a arbitriry large number
    def __init__(self, stream_limit = 20):
        self.stream_limit = stream_limit 
    
    def place(self, soc: SoCBase, acc: "accelerator name", configs: List[Tuple["lat", "resource"]]):
        limits = soc.get_resource_limit(acc)
        stream_limit = min(limits['num_stream'], self.stream_limit)
        resource_limit = limits['num_pes']
        
        ids = sorted(range(len(configs)), key=lambda x: configs[x][0])
        
        id2stream = {}
        id2time = {}
        
        min_lat = 0
        total_lat = 0
        cur_pe_usage = 0
        i = 0
        stream_id = -1
        attrs = {'n_increase': 0}
        for i in range(len(ids)):
            id = ids[i]
            pe_usage = configs[id][1]
            if cur_pe_usage + pe_usage > resource_limit or stream_id + 1 >= stream_limit:
                min_lat += configs[ids[i-1]][0]
                cur_pe_usage = 0
                stream_id = 0
                attrs['n_increase'] += 1
            else:
                cur_pe_usage += pe_usage
                stream_id += 1
            id2stream[id] = stream_id 
            id2time[id] = min_lat 
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
    mapper = EvolutionMapper(placer)
    cg = ComputationGraph(graph, mapper)
    accs = [[MaestroNVDLA("MaestroNVDLA(0)", 2), MaestroDepthwiseShiDianNao("ShiDianNao(1)"), MaestroGemmTPU("MaestroGemmTPU")]]
    soc = MeshSoC(accs)
    mapper.fuzz_test(soc)
    # complete_time = cg.map(soc)
    # soc.report()
    # cg.visualize_packing(soc, complete_time)
    # cg.visualize('_'.join(models))
    # print("compute lowerbound is ", cg.lower_bound(soc))
    # print("compute uses ", complete_time)
    
if __name__ == "__main__":
    random.seed(3)
    os.system("mkdir -p .cache")
    os.system("mkdir -p pics")
    MaestroAcceleratorBase.load_cache()
    main()
    MaestroAcceleratorBase.store_cache()
