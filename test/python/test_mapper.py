from typing import Dict, Any, List, Set, Optional, Union, Tuple
import networkx as nx 
import os 
from itertools import combinations, product
import math
import random


from domino.utils import ONNXConvertor
from domino.graph_pass import set_graph_precision, GraphPrinter, GraphVisitor
from domino.graph_ir import Op, SubGraph, Graph, Tensor
import matplotlib.pyplot as plt


class Accelerator:
    '''
    Init the attributes before each mapping.
    '''
    def init(self):
        self.last_complete_time = 0
        self.logs = []
        
    '''
    Get all accelerators for this accelerator instance
    Return: List[Accelerator]
    '''
    def get_accs(self):
        raise NotImplementedError
    
    '''
    Time to execute op on this accelerator. 
    '''
    def time(self, op: Op.NamedOp) -> float:
        raise NotImplementedError
    
    '''
    Commit op to this accelerator.
    '''
    def commit(self, op: Op.NamedOp, start_time: float, exec_time: float):
        self.logs.append([op, start_time, exec_time])
        self.last_complete_time = start_time + exec_time 

'''
Driver class for mapping. 
'''
class ComputationGraph:
    def __init__(self, graph, mapper_type, verbose = False):
        self.g = graph
        if mapper_type == 'greedy':
            self.mapper = GreedyMapper(self, verbose)
        else:
            raise RuntimeError
        self.verbose = verbose 
    def reset(self):
        for u in self.g.nodes:
            self.g.nodes[u]['committed'] = False 
    
    def visualize(self, filepath):
        acc2color = {}
        s = 'digraph G{\n'
        for id in self.g.nodes:
            u = self.g.nodes[id]
            if u['acc'] not in acc2color:
                acc2color[u['acc']] = f'{hex(random.randint(0,256*256*256))}'
            op_type = u['op'].name
            c = acc2color[u['acc']]
            s += f'node{id} [label=\"{op_type}\", color=\"#{c}\"]\n'
        for u,v in self.g.edges:
            s += f'node{u} -> node{v}\n'
        s += '}\n'
        
        with open('tmp.gv', 'w') as f:
            f.write(s)
        
        cmd = f'dot -Tpng tmp.gv -o {filepath}.png'
        os.system(cmd)
    
    def check(self):
        acc2busyPeriod = {}
        for nid in self.g.nodes:
            node = self.g.nodes[nid]
            if not node['committed']:
                if self.verbose:
                    print(f'{node} not committed')
                return False
            for i_nid in self.g.pred[nid]:
                if self.g.nodes[i_nid]['end'] > node['start']:
                    if self.verbose:
                        op = node['op']
                        oopp = self.g.nodes[i_nid]['op']
                        print(f'time violated {oopp.name}{i_nid} -> {op.name}{nid}')
                    return False 
            acc = node['acc']
            if acc not in acc2busyPeriod:
                acc2busyPeriod[acc] = []
            for r in acc2busyPeriod[acc]:
                if not (r[0] >= node['end'] or r[1] <= node['start']):
                    if self.verbose:
                        op = node['op']
                        oopp = r[2]['op']
                        print(f'acc violated {op.name}{nid} -> {acc} {r[0]} {r[1]} {oopp.name}{r[3]}')
                    return False
            acc2busyPeriod[acc].append([node['start'], node['end'], node, nid])
        return True
                        
    def map(self, accs: List[Accelerator]):
        print(self.g)
        self.reset()
        fine_grained_accs = []
        for acc in accs:
            fine_grained_accs += acc.get_accs()
        print ("accelerators:")
        for acc in fine_grained_accs:
            print(acc)
        self.mapper(fine_grained_accs)
        assert (self.check())
        
class MapperBase:
    # the complete time if we map op to acc 
    def __init__(self, cg: ComputationGraph, verbose: bool = False):
        self.cg = cg  
        self.g = nx.DiGraph()
        self.verbose = verbose 
    
    '''
    Time the op at acc whose last job finishes at complete_time
    '''
    def timing(self, op: int, acc: Accelerator, complete_time: float) -> Tuple[float, float]:
        start_time = complete_time
        for input_id in self.cg.g.pred[op]:
            assert self.cg.g.nodes[input_id]['committed']
            start_time = max(start_time, self.cg.g.nodes[input_id]['end'])
        finish_time = start_time + acc.time(self.g.nodes[op]['op'])
        return start_time, finish_time
    
    '''
    Commit a bunch of ops. 
    '''
    def commit(self, accs: Tuple[Accelerator], ops: List[int], start_times: List[float], exec_times: List[float]):
        for acc, id, start_time, exec_time in zip(accs, ops, start_times, exec_times):
            if self.verbose: 
                name = self.cg.g.nodes[id]['op'].name
                print(f'bind {name}{id} to {acc}, {start_time}, {exec_time}')
            acc.commit(id, start_time, exec_time)
            self.cg.g.nodes[id]['acc'] = acc 
            self.cg.g.nodes[id]['committed'] = True 
            self.cg.g.nodes[id]['start'] = start_time 
            self.cg.g.nodes[id]['end'] = start_time + exec_time 
            self.g.remove_node(id)
    
    '''
    The mapping implementation. The mapper should annotate the start/end/acc attributes for each node.
    '''
    def __call__(self, accs: List[Accelerator]):
        raise NotImplementedError
 
class GreedyMapper(MapperBase):
    def __init__(self, cg: ComputationGraph, verbose: bool): 
        super(GreedyMapper, self).__init__(cg, verbose)

    def __call__(self, accs: List[Accelerator]):
        self.g = self.cg.g.copy() 
        # an matching example 
        for acc in accs: 
            acc.init()
        
        while len(self.g.nodes):
            nodes = [node for node in self.g.nodes if not self.g.pred[node]]
            candidates = product(accs, repeat=len(nodes))
            min_time = math.inf
            best_candidate: Tuple[Accelerator] = tuple()
            candidate_start_times: List[float] = []
            candidate_exec_times: List[float] = []
            for i, candidate in enumerate(candidates):
                # print (f"run candidate {i}")
                start_times = []
                exec_times = []
                complete_times = {acc:acc.last_complete_time for acc in accs}
                for id,acc in zip(nodes, candidate):
                    start,finish = self.timing(id, acc, complete_times[acc])
                    complete_times[acc] = finish
                    start_times.append(start)
                    exec_times.append(finish - start)
                complete_time = max(complete_times.values())
                if min_time > complete_time:
                    min_time = complete_time
                    best_candidate = candidate 
                    candidate_start_times = start_times 
                    candidate_exec_times = exec_times  
            self.commit(best_candidate, nodes, candidate_start_times, candidate_exec_times)

class StreamAcc(Accelerator):
    cnt = 0
    def __init__(self, father):
        self.father = father 
        self.id = StreamAcc.cnt 
        StreamAcc.cnt += 1
    def time(self, op: Op.NamedOp) -> float:
        if op.name == Op.OpName.ConvOp:
            return 1.0
        elif op.name == Op.OpName.MatrixOp:
            return 2.0 
        return 0.1   
    def __str__(self):
        return f'StreamAcc{self.id}'        
        

class GPUAcc(Accelerator):
    def __init__(self, n_stream = 2):
        self.n_stream = n_stream 
        self.streams = []
        for i in range(n_stream):
            self.streams.append(StreamAcc(self))
    def get_accs(self):
        return self.streams
       
'''
A converter to transform graph ir into networkx graph
'''
class GraphIRConverter(GraphVisitor):
    def __init__(self):
        super(GraphIRConverter, self).__init__()
        
    def get_id(self, op: Op.NamedOp):
        if op not in self.op2index:
            self.op2index[op] = len(self.op2index)
        return self.op2index[op]
    def visit_op(self, op: Op.NamedOp, boundary_tensors: Set[Tensor]):
        if self.has_visited_op(op):
            return self.get_op_visited(op)
        id = self.get_id(op)
        self.g.add_node(id, op=op, committed = False, acc = Accelerator(), start=0.0, end = 0.0)
        for name, input_tensor in op.inputs.items():
            if input_tensor in boundary_tensors:
                # subgraph inputs
                pass
            elif input_tensor.produce_op is not None:
                # compute op
                input_id = self.get_id(input_tensor.produce_op)
                self.g.add_edge(input_id, id)                
                visitor = self.get_visitor(input_tensor.produce_op)
                visitor(input_tensor.produce_op, boundary_tensors)
            else:
                # producer op
                pass
        return self.record_visited_op(op, None)
    def __call__(self, graph: Graph, specify_subgraphs: Optional[Union[Set[str], List[str]]] = None, init_state=True) -> Any:
        self.op2index = {}
        self.g = nx.DiGraph()
        self.visit_graph(graph, specify_subgraphs=specify_subgraphs, init_state=init_state)
        return self.g

def get_graph(path: str):
    convertor = ONNXConvertor(path, inference=True)
    graph = convertor.parse()
    return GraphIRConverter()(graph)


def visualize(graph, name = 'g'):
    s = 'digraph G{\n'
    for u,op in graph.nodes.data('op'):
        s += f'node{u} [label=\"{op.name}\"]\n'
    for u,v in graph.edges:
        s += f'node{u} -> node{v}\n'
    s += '}\n'
    
    with open('tmp.gv', 'w') as f:
        f.write(s)
    
    cmd = f'dot -Tpng tmp.gv -o {name}.png'
    os.system(cmd)
        

def main():
    # config_path = "new_resnet18_pareto.json"
    # model_path = "raw_resnet18.onnx"
    # config_path = "new_mobilenetv2_pareto.json"
    # model_path = "raw_mobilenetv2.onnx"
    # config_path = "new_resnet50_pareto.json"
    # model_path = "raw_resnet50.onnx"
    model_path = "yolov5s_640x640.simplify.onnx"
    graph = get_graph(model_path)
    print(graph)
    visualize(graph)
    cg = ComputationGraph(graph, 'greedy', True)
    accs:List[Accelerator] = [GPUAcc(1), GPUAcc(2), GPUAcc(3)]
    cg.map(accs)
    cg.visualize('./yolo_mapped.png')
    

if __name__ == "__main__":
    main() 