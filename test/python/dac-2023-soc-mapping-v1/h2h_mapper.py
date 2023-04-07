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
from domino.base import MaestroAcceleratorBase, AccTask, AccStream, SoCBase
from domino.accelerator import MaestroConvAccelerator, MeshSoC, MaestroNVDLA, MaestroGemmTPU, MaestroDepthwiseShiDianNao
from domino.program_ir import ConstInt, ConstUInt, ConstFloat, ConstString, ExprList
import matplotlib.pyplot as plt
from domino import global_timer

from base import ComputationGraph, MapperBase, GraphIRConverter, get_graph, visualize, ParallelMachineScheduler, GreedyScheduler

class GreedyMapper(MapperBase):
    def __init__(self, scheduler: ParallelMachineScheduler, verbose: bool = False): 
        self.scheduler = scheduler 
        super(GreedyMapper, self).__init__(verbose)
    
    def __call__(self, soc: SoCBase):
        assert self.scheduler is not None
        self.g = self.cg.g.copy()
        self.soc = soc
        
        machines = soc.get_machines()
        resource_constraint = soc.get_all_resource_limit() 
        
        
        pred_cnt = {x:len(self.g.pred[x]) for x in self.g.nodes}
        frontiers = [x for x, cnt in pred_cnt.items() if cnt == 0]
        
        accs = soc.get_all_accs()
        compute_times = self.get_compute_time(accs)
        resource_usages = self.get_resource_usage(accs)

        while len(frontiers):
            timing = []          
            for x in frontiers:
                acc_timing = {}
                for acc in accs[MapperBase.op2task[self.cg.g.nodes[x]['op'].name]]:
                    compute_time = compute_times[x][acc]
                    comm_time = soc.eval_communication(
                        (acc, self.cg.g.nodes[x]['task']),
                        [(self.cg.g.nodes[pred]['acc'][0], self.cg.g.nodes[pred]['task']) for pred in self.cg.g.pred[x]]
                    )
                    acc_timing[acc] = compute_time + comm_time
                timing.append(acc_timing)
            resource_usage = [resource_usages[x] for x in frontiers]  
            _, placement, _, _ = self.scheduler.schedule(2, machines, resource_constraint, resource_usage, timing)
            
            candidate = tuple((placement[i][0], placement[i][1][0]) for i in range(len(frontiers)))
            elapsed_time = self.commit(soc, frontiers, candidate, False)
            
            l = len(frontiers)
            for i in range(l):
                x = frontiers[i]
                for succ in self.cg.g.succ[x]:
                    pred_cnt[succ] -= 1
                    if pred_cnt[succ] == 0:
                        frontiers.append(succ)
            frontiers = frontiers[l:]
        return elapsed_time 
def get_graph(path: str):
    convertor = ONNXConvertor(path, inference=True)
    graph = convertor.parse()
    return GraphIRConverter()(graph)
        
def test1():
    N,K,H,W,P,Q,C,R,S = 1,128,256,256,256,256,64,3,3
    i1 = Tensor(shape=[N,C,H,W], dtype = "float32")
    w1 = Tensor(shape = [K,C,R,S], dtype="float32")
    o1 = Tensor(shape=[N,K,P,Q],dtype="float32")
    i2 = Tensor(shape=[N,C,H,W], dtype = "float32")
    w2 = Tensor(shape = [K,C,R,S], dtype="float32")
    o2 = Tensor(shape=[N,K,P,Q],dtype="float32")
    
    conv1 = Op.NamedOp(Op.OpName.ConvOp.Conv2d, {'inputs':i1, 'weight': w1}, {'output':o1}, attrs = {'strides': Attribute(ExprList([ConstInt(1), ConstInt(1)]))})
    conv2 = Op.NamedOp(Op.OpName.ConvOp.Conv2d, {'inputs':i2, 'weight': w2}, {'output':o2}, attrs = {'strides': Attribute(ExprList([ConstInt(1), ConstInt(1)]))})
    g = nx.DiGraph()
    
    g.add_node(0, op = conv1, task = AccTask("T0", "Conv2d", conv1.get_config()), start = 0.0, finish = 0.0, acc = None)
    g.add_node(1, op = conv2, task = AccTask("T1", "Conv2d", conv2.get_config()), start = 0.0, finish = 0.0, acc = None)
    cg = ComputationGraph(g, 'greedy', True)
    accs = []
    for i in range(2):
        accs.append([])
        for j in range(2):
            acc = MaestroNVDLA(f"MaestroNVDLA({i},{j})") 
            accs[-1].append(acc)
    soc = MeshSoC(accs)
    cg.map(soc)
    

    
def main():
    models = ['resnet18']
    graph = get_graph(models)
    print(graph)
    visualize(graph)
    mapper = GreedyMapper(True)
    cg = ComputationGraph(graph, mapper, True)
    accs = [[MaestroNVDLA("MaestroNVDLA(0)"), MaestroDepthwiseShiDianNao("ShiDianNao(1)"), MaestroGemmTPU("TPU(2)")]]
    soc = MeshSoC(accs)
    cg.map(soc)
    cg.visualize('mobilenet')
    

if __name__ == "__main__":
    os.system("mkdir -p .cache")
    MaestroAcceleratorBase.load_cache()
    main() 
    MaestroAcceleratorBase.store_cache()