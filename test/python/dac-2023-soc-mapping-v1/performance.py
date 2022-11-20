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
from domino.accelerator import ConvAccelerator, MeshSoC, NVDLA, GemmTPU, DepthwiseShiDianNao, ConvShiDianNao, GemmNVDLA
from domino.program_ir import ConstInt, ConstUInt, ConstFloat, ConstString, ExprList
import matplotlib.pyplot as plt
from domino import global_timer

from base import GraphIRConverter, ComputationGraph, get_graph, MapperBase, visualize, GreedyScheduler
from evolution_mapper import SimplePlacer, EvolutionMapper
from h2h_mapper import GreedyMapper 

graphs = {
    'vision1': [('resnet18', 2), ('resnet50', 2), ('yolo', 2), ('GoogLeNet', 2)],
    'vision2': [('resnet18', 8), ('resnet50', 8), ('yolo', 8), ('GoogLeNet', 8)],
    'nlp1': [('bert', 2), ('lstm', 2)],
    'nlp2': [('bert', 4), ('lstm', 4)],
    'mixed1': [('GoogLeNet', 2), ('yolo', 2), ('bert', 2), ('lstm', 2), ('mobilenet', 2)],
    'mixed2': [('GoogLeNet', 4), ('yolo', 4), ('bert', 4), ('lstm', 2), ('mobilenet', 4)],
    'mixed3': [('GoogLeNet', 8), ('yolo', 8), ('bert', 8), ('lstm', 2), ('mobilenet', 8)]
}

socs = {
    'SmallSoC': {
        'accelerator_matrix': [
            [NVDLA('NVDLA(0,0)', 2), NVDLA('NVDLA(0,1)', 2)],
            [DepthwiseShiDianNao("ShiDianNao(1,0)"), GemmTPU("GemmTPU(1,1)", 2)]
        ],
        'name': 'SmallSoC'
} , 'SmallSoC-GEMM': {
        'accelerator_matrix': [
            [GemmNVDLA('NVDLA(0,0)', 2), GemmNVDLA('NVDLA(0,1)', 2)],
            [DepthwiseShiDianNao("ShiDianNao(1,0)"), GemmTPU("GemmTPU(1,1)", 2)]
        ],
        'name': 'SmallSoC'
} , 'LargeSoC': {
        'accelerator_matrix': [
            [DepthwiseShiDianNao("ShiDianNao(0,0)"), DepthwiseShiDianNao("ShiDianNao(0,1)")],
            [NVDLA('NVDLA(1,0)', 2), NVDLA('NVDLA(1,1)', 2)],
            [NVDLA('NVDLA(2,0)', 2), NVDLA('NVDLA(2,1)', 2)],
            [GemmTPU('GemmTPU(3,0)', 2), GemmTPU('GemmTPU(3,1)', 2)],
        ],
        'name': 'LargeSoC'
} , 'LargeSoC-GEMM': {
        'accelerator_matrix': [
            [DepthwiseShiDianNao("ShiDianNao(0,0)"), DepthwiseShiDianNao("ShiDianNao(0,1)")],
            [GemmNVDLA('NVDLA(1,0)', 2), GemmNVDLA('NVDLA(1,1)', 2)],
            [GemmNVDLA('NVDLA(2,0)', 2), GemmNVDLA('NVDLA(2,1)', 2)],
            [GemmTPU('GemmTPU(3,0)', 2), GemmTPU('GemmTPU(3,1)', 2)],
        ],
        'name': 'LargeSoC'
}
}

def run(alg: str, model_tag: str, soc_tag: str, bandwidth: str, verbose = False, cached = False):
    assert alg in ['H2H', 'COMB', 'MAGMA']
    if 'nlp' in model_tag and 'GEMM' not in soc_tag:
        soc_tag += "-GEMM"
    print (f'running {alg} {model_tag} {soc_tag} {bandwidth}')
    models = []
    for network, batch_size in graphs[model_tag]:
        models += [network] * batch_size
    graph = get_graph(models)
    if verbose: 
        visualize(graph, f'pics/{model_tag}')
    print(f'Graph: {models}', graph)
    if alg == 'H2H':
        scheduler = GreedyScheduler(resource_limit=[(0,1)])
        mapper = GreedyMapper(scheduler = scheduler, verbose = verbose)
    elif alg == 'COMB':
        file_path = '.cache/'+'_'.join([alg, model_tag, soc_tag, bandwidth]) + '.pkl'
        scheduler = GreedyScheduler()
        mapper = EvolutionMapper(scheduler = scheduler, verbose = verbose, file_path=file_path, cached = cached)
    elif alg == 'MAGMA':
        file_path = '.cache/'+'_'.join([alg, model_tag, soc_tag, bandwidth]) + '.pkl'
        scheduler = GreedyScheduler(resource_limit=[(0,1)]) # constraint the stream usage
        mapper = EvolutionMapper(scheduler = scheduler, verbose = verbose, file_path = file_path, cached = cached)
    else:
        raise RuntimeError(f'unknow alg {alg}')    
    cg = ComputationGraph(graph, mapper, verbose)
    soc_args = copy.deepcopy(socs[soc_tag])
    if bandwidth == 'highBW':
        soc_args['off_chip_nearest_bw'] = 3.2
    elif bandwidth == 'lowBW':
        soc_args['off_chip_nearest_bw'] = 0.8
    soc = MeshSoC(**soc_args)
    if verbose: 
        print("compute lowerbound is ", cg.lower_bound(soc))
        print (f'soc with accs: {soc.accelerator_graph.nodes}')
    complete_time = cg.map(soc)
    # scheduler.store('./schedules.pkl')
    energy_consumption = soc.get_current_energy_consumption()
    filename = '_'.join([alg, model_tag, soc_tag, bandwidth])
    if verbose: 
        cg.visualize_packing(soc, complete_time, filepath = f'pics/{filename}')
        cg.visualize(f'pics/{filename}')
    print(f'--------alg:{alg}|model:{model_tag}|SoC:{soc_tag}|BW:{bandwidth}----------')
    soc.report()
    os.system('mkdir -p pe_curve')
    soc.store_pe_curve(f'pe_curve/{filename}.pkl')
    profile = soc.profile()
    os.system("mkdir -p profile")
    with open(f'profile/{filename}.pkl', 'wb') as f:
        pkl.dump(profile, f)
    # if alg == 'COMB':
    #     mapper.export(filename)
    print("compute uses ", complete_time, 'energy: ', energy_consumption)
    return complete_time, energy_consumption

def main(algs, models, socs, bandwidths, store_path = "res", cached = False):
    results = {}
    failed = []
    for model_tag in models:
        for soc_tag in socs:
            for alg in algs:
                for bw in bandwidths:
                    # try: 
                    key = '_'.join([model_tag, soc_tag, alg])
                    global_timer.start(key)
                    results[(model_tag, soc_tag, alg, bw)] = run(alg, model_tag, soc_tag, bw, cached = cached)
                    global_timer.stop(key)
                    # except:
                    #     print (f'[WARNING] run{alg}, {model_tag}, {soc_tag}, {bw} failed')
                    #     failed.append((alg, model_tag, soc_tag, bw))
    with open(f'./result/{store_path}.pkl', 'wb') as f:
        pkl.dump(results, f)
    for k, v in results.items():
        print(k, v)
    iter = 0
    # while failed:
    #     new_failed = []
    #     for config in failed:
    #         try: 
    #             results[config] = run(*config)
    #         except:
    #             new_failed.append(config)
    #     failed = new_failed
    #     iter += 1 
    #     for k, v in results:
    #         print(k, v)
    #     with open(f'./result/{store_path}.pkl', 'wb') as f:
    #         pkl.dump(results, f)
    
    print("failed: ", failed)
    global_timer.show()

import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type = str, nargs = '+', default=['COMB', 'H2H', 'MAGMA'], choices = ['COMB', 'H2H', 'MAGMA'])
    parser.add_argument('--model', type = str, nargs = '+',default=graphs.keys())
    parser.add_argument('--soc', type = str, nargs = '+', default=socs.keys(), choices = socs.keys())
    parser.add_argument('--bandwidth', type = str, nargs = '+', default=['highBW'], choices = ['highBW', 'lowBW'])
    parser.add_argument('--store_path', type = str, default = "res")
    parser.add_argument('--cached', action = "store_true")
    args = parser.parse_args()
    
    random.seed(1)
    os.system("mkdir -p .cache")
    os.system("mkdir -p pics")
    os.system("mkdir -p result")
    AcceleratorBase.load_cache()
    main(args.alg, args.model, args.soc, args.bandwidth, args.store_path, args.cached)
    # run('COMB', 'vision', 'LargeSoC', True)
    # main()
    AcceleratorBase.store_cache()
              