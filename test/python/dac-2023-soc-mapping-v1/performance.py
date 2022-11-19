from typing import Dict, Any, List, Set, Optional, Union, Tuple
import networkx as nx 
import os 
from itertools import combinations, product
import math
import random
import time 
import copy
import numpy as np
import pickle as pkl

from domino.utils import ONNXConvertor
from domino.graph_pass import set_graph_precision, GraphPrinter, GraphVisitor
from domino.graph_ir import Op, SubGraph, Graph, Tensor, Attribute
from domino.base import AcceleratorBase, AccTask, AccStream, SoCBase
from domino.accelerator import ConvAccelerator, MeshSoC, NVDLA, GemmTPU, DepthwiseShiDianNao, ConvShiDianNao
from domino.program_ir import ConstInt, ConstUInt, ConstFloat, ConstString, ExprList
import matplotlib.pyplot as plt
from domino import global_timer

from base import GraphIRConverter, ComputationGraph, get_graph, MapperBase, visualize
from evolution_mapper import SimplePlacer, EvolutionMapper, GreedyScheduler
from h2h_mapper import GreedyMapper 

graphs = {
    'vision': [('resnet18', 2), ('resnet50', 2), ('yolo', 4), ('GoogLeNet', 4)],
    'mixed': [('resnet18', 2), ('resnet50', 2), ('mobilenet', 4), ('GoogLeNet', 2), ('yolo', 2)],
    'nlp': [('ViT', 4), ('Bert', 4), ('GPT2', 2)],
    # 'mixed': [('GoogLeNet', 4), ('yolo', 2), ('Bert', 2)]
}

socs = {
    'SmallSoC': MeshSoC(
        [
            [NVDLA('NVDLA(0,0)', 2), NVDLA('NVDLA(0,1)', 2)],
            [DepthwiseShiDianNao("ShiDianNao(1,0)"), GemmTPU("GemmTPU(1,1)")]
        ],
        name = 'SmallSoC'
) , 'LargeSoC': MeshSoC(
        [
            [DepthwiseShiDianNao("ShiDianNao(0,0)"), DepthwiseShiDianNao("ShiDianNao(0,1)")],
            [NVDLA('NVDLA(1,0)', 2), NVDLA('NVDLA(1,1)', 2)],
            [NVDLA('NVDLA(2,0)', 2), NVDLA('NVDLA(2,1)', 2)],
            [DepthwiseShiDianNao('ConvShiDianNao(1,0)', 2), GemmTPU('GemmTPU(1,1)', 2)],
        ],
        name = 'LargeSoC'
)
}

def run(alg: str, model_tag: str, soc_tag: str, verbose = False):
    assert alg in ['H2H', 'COMB', 'MAGMA']
    print (f'running {alg} {model_tag} {soc_tag}')
    models = []
    for network, batch_size in graphs[model_tag]:
        models += [network] * batch_size
    graph = get_graph(models)
    if verbose: 
        visualize(graph, f'pics/{model_tag}')
    print(f'Graph: {models}', graph)
    if alg == 'H2H':
        mapper = GreedyMapper(verbose)
    elif alg == 'COMB':
        # placer = SimplePlacer()
        scheduler = GreedyScheduler()
        mapper = EvolutionMapper(scheduler = scheduler, verbose = verbose)
    elif alg == 'MAGMA':
        placer = SimplePlacer(1)
        mapper = EvolutionMapper(placer, verbose = verbose)
    else:
        raise RuntimeError(f'unknow alg {alg}')    
    cg = ComputationGraph(graph, mapper, verbose)
    soc = copy.deepcopy(socs[soc_tag])
    print("compute lowerbound is ", cg.lower_bound(soc))
    print (f'soc with accs: {soc.accelerator_graph.nodes}')
    complete_time = cg.map(soc)
    filename = '_'.join([alg, model_tag, soc_tag])
    if verbose: 
        print(f'--------alg:{alg}|model:{model_tag}|SoC:{soc_tag}----------')
        soc.report()
        cg.visualize_packing(soc, complete_time, filepath = f'pics/{filename}')
        cg.visualize(f'pics/{filename}')
    print (filename)
    print("compute uses ", complete_time)
    return complete_time

def main():
    algs = ['COMB', 'H2H', 'MAGMA']
    results = {}
    failed = []
    for model_tag in graphs:
        for soc_tag in socs:
            for alg in algs:
                # try: 
                key = '_'.join([model_tag, soc_tag, alg])
                global_timer.start(key)
                results[(model_tag, soc_tag, alg)] = run(alg, model_tag, soc_tag)
                global_timer.stop(key)
                # except:
                #     print (f'[WARNING] run{alg}, {model_tag}, {soc_tag} failed')
                #     failed.append((alg, model_tag, soc_tag))
    print (results)
    with open('res.pkl', 'wb') as f:
        pkl.dump(results, f)
    print("failed: ", failed)
    global_timer.show()

if __name__ == "__main__":
    random.seed(1)
    os.system("mkdir -p .cache")
    os.system("mkdir -p pics")
    AcceleratorBase.load_cache()
    run('COMB', 'vision', 'LargeSoC', True)
    # main()
    AcceleratorBase.store_cache()
              