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
from domino.accelerator import ConvAccelerator, MeshSoC, NVDLA
from domino.program_ir import ConstInt, ConstUInt, ConstFloat, ConstString, ExprList
import matplotlib.pyplot as plt
from domino import global_timer


'''
Driver class for mapping. 
'''
class ComputationGraph:
    def __init__(self, graph, mapper_type, verbose = False):
        self.g = graph
        if mapper_type == 'greedy':
            self.mapper = GreedyMapper(self, verbose)
        else:
            raise RuntimeError(f"{self.mapper} unknown")
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

    def map(self, soc: SoCBase):
        self.reset()
        self.mapper(soc)
        # assert (self.check())

class MapperBase:
    # the complete time if we map op to acc 
    def __init__(self, cg: ComputationGraph, verbose: bool = False):
        self.cg = cg  
        self.g = nx.DiGraph()
        self.verbose = verbose 
    
    '''
    Commit a bunch of ops. 
    '''
    def commit(self, soc: SoCBase, nodes: List[int], streams: Tuple[Tuple[AcceleratorBase, int]], simulate: bool = False):
        current_time = soc.elapsed_time
        assert len(nodes) == len(streams)
        global_timer.start('push_task')
        for id, stream in zip(nodes, streams):
            soc.push_task(self.cg.g.nodes[id]['task'], *stream)
        global_timer.stop('push_task')
        
        global_timer.start('commit_all_tasks')
        complete_time = soc.commit_all_tasks() 
        global_timer.stop('commit_all_tasks')
        if not simulate:
            for id, stream in zip(nodes, streams):
                if self.verbose:
                    print(f'Bind {id} to {stream} at {current_time} to {complete_time}')
                node = self.cg.g.nodes[id]
                node['start'] = current_time 
                node['finish'] = complete_time
                node['acc'] = stream
                self.g.remove_node(id)
        return complete_time
    
    '''
    The mapping implementation. The mapper should annotate the start/end/acc attributes for each node.
    '''
    def __call__(self, soc: SoCBase):
        raise NotImplementedError()

class GreedyMapper(MapperBase):
    def __init__(self, cg: ComputationGraph, verbose: bool): 
        super(GreedyMapper, self).__init__(cg, verbose)

    def __call__(self, soc: SoCBase):
        streams = soc.get_all_streams()
        
        self.g = self.cg.g.copy() 
        # an matching example 
        iter = 0
        while len(self.g.nodes):
            nodes = [node for node in self.g.nodes if not self.g.pred[node]]
            candidates = product(streams, repeat=len(nodes))
            print ('nodes:', nodes)
            print ('n_candidate: ', len(streams) ** len(nodes))
            min_time = math.inf
            best_candidate: Tuple[Tuple[AcceleratorBase, int]] = tuple()
            for candidate in candidates:
                # print (f"run candidate {i}")
                complete_time = self.commit(soc.snapshot(), nodes, candidate, True)
                print (f'simulating candidate: ', candidate, complete_time)
                if min_time > complete_time:
                    min_time = complete_time
                    best_candidate = candidate 
            self.commit(soc, nodes, best_candidate)
            global_timer.show(f'-----------------iter{iter}----------------')
            iter += 1
       
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
        self.g.add_node(id, op=op, task = AccTask(id), start = 0.0, end = 0.0, acc = (None, None))
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
        self.g = nx.transitive_closure(self.g)
        considered_ops = [Op.OpName.ConvOp.Conv2d, Op.OpName.ConvOp.DepthwiseConv2d, Op.OpName.MatrixOp.Gemm]
        self.g = self.g.subgraph([id for id in self.g.nodes if self.g.nodes[id]['op'].name in considered_ops])
        G = nx.transitive_reduction(self.g)
        G.add_nodes_from(self.g.nodes(data = True))
        self.g = G
        
        for id in self.g.nodes:
            node = self.g.nodes[id]
            print(id, node)
            task = node['task']
            task.name = f'T{id}'
            task.depend_tasks = [self.g.nodes[i]['task'] for i in self.g.pred[id]]
            task.params = node['op'].get_config()
            print (task.params)
            if node['op'].name == Op.OpName.ConvOp.Conv2d:
                print("Conv2dOp", node['op'], type(node['op']))
                task.task_kind = "Conv2d"
            elif node['op'].name == Op.OpName.ConvOp.DepthwiseConv2d:
                print("DepthWise", node['op'])
                task.task_kind = "DepthWise" 
            elif node['op'].name == Op.OpName.MatrixOp.Gemm:
                print("Gemm", node['op'])
                task.task_kind = "Gemm"
            else:
                raise RuntimeError()
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
    accs = []
    for i in range(2):
        accs.append([])
        for j in range(2):
            acc = NVDLA(f"NVDLA({i},{j})") 
            accs[-1].append(acc)
    soc = MeshSoC(accs)
    cg.map(soc)
    cg.visualize('./yolo_mapped.png')
    

if __name__ == "__main__":
    os.system("mkdir -p .cache")
    AcceleratorBase.load_cache()
    main() 
    AcceleratorBase.store_cache()