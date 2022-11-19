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

from base import ComputationGraph, MapperBase, GraphIRConverter, get_graph, visualize

class GreedyMapper(MapperBase):
    def __init__(self, verbose: bool = False): 
        super(GreedyMapper, self).__init__(verbose)

    def __call__(self, soc: SoCBase):
        accs = soc.get_all_accs()
        op2task = {Op.OpName.ConvOp.Conv2d: "Conv2d", Op.OpName.ConvOp.DepthwiseConv2d: "Depthwise", Op.OpName.MatrixOp.Gemm:"Gemm"}
                
        self.g = self.cg.g.copy() 
        # an matching example 
        iter = 0
        elapsed_time = 0
        while len(self.g.nodes):
            nodes = [node for node in self.g.nodes if not self.g.pred[node]]
            for node in nodes:
                opname = self.g.nodes[node]['op'].name
            candidates = list(product(*[accs[op2task[self.g.nodes[node]['op'].name]] for node in nodes]))
            
            min_time = math.inf
            best_candidate: Tuple[Tuple[AcceleratorBase, int]] = tuple()
            for candidate in random.sample(candidates, k=min(100, len(candidates))):
                candidate = tuple((acc, 0) for acc in candidate)
                # print (f"run candidate {i}")
                global_timer.start('simulate')
                complete_time = self.commit(soc.snapshot(), nodes, candidate, True)
                global_timer.stop('simulate')
                if min_time > complete_time:
                    min_time = complete_time
                    best_candidate = candidate 
            elapsed_time = self.commit(soc, nodes, best_candidate)
            if self.verbose:
                global_timer.show(f'-----------------iter{iter}----------------')
            iter += 1
        global_timer.show(f'finished mapping with time {elapsed_time}') 
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
            acc = NVDLA(f"NVDLA({i},{j})") 
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
    accs = [[NVDLA("NVDLA(0)"), DepthwiseShiDianNao("ShiDianNao(1)"), GemmTPU("TPU(2)")]]
    soc = MeshSoC(accs)
    cg.map(soc)
    cg.visualize('mobilenet')
    

if __name__ == "__main__":
    os.system("mkdir -p .cache")
    AcceleratorBase.load_cache()
    main() 
    AcceleratorBase.store_cache()