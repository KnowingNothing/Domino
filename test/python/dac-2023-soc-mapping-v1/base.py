from typing import Dict, Any, List, Set, Optional, Union, Tuple
import networkx as nx 
import os 
from itertools import combinations, product
from functools import reduce
import math
import random
import time 
from PIL import Image, ImageDraw
import numpy as np
import pickle as pkl

from domino.utils import ONNXConvertor
from domino.graph_pass import set_graph_precision, GraphPrinter, GraphVisitor
from domino.graph_ir import Op, SubGraph, Graph, Tensor, Attribute
from domino.base import AcceleratorBase, AccTask, AccStream, SoCBase
from domino.accelerator import ConvAccelerator, MeshSoC, NVDLA, GemmTPU, DepthwiseShiDianNao
from domino.program_ir import ConstInt, ConstUInt, ConstFloat, ConstString, ExprList
import matplotlib.pyplot as plt
from domino import global_timer

class ParallelMachineScheduler:
    def __init__(self):
        self.__cache = []
    def schedule(self, 
                num_resources: int, 
                machines: List[str], 
                resource_costraint: Dict[str, List[float]], 
                resource_requirement: List[Dict[str, List[float]]], 
                timing: List[Dict[str, float]]):
        raise NotImplementedError()
    def regsiter(self, data):
        self.__cache.append(data)
    def store(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self.__cache, f)

class GreedyScheduler(ParallelMachineScheduler):
    def __init__(self, resource_limit:Union[None, List[Tuple[int, Any]]] = None):
        super(GreedyScheduler, self).__init__()
        self.resource_limit = resource_limit 

    def __str__(self):
        return f'Greedy_Scheduler(resource_limit:{self.resource_limit})'
    
    def schedule(self,  
                num_resources: int, 
                machines: List[str], 
                resource_costraint: Dict[str, List[float]], 
                resource_requirement: List[Dict[str, List[float]]], 
                timing: List[Dict[str, float]]):
        self.regsiter((num_resources, machines, resource_costraint, resource_requirement, timing))

        if self.resource_limit is not None:
            for rid, limit in self.resource_limit: 
                for acc in resource_costraint:
                    resource_costraint[acc][rid] = min(resource_costraint[acc][rid], limit)
        
        resource_used = {machine: [0] * num_resources for machine in machines} # 
        machine_commit_time = {machine: 0 for machine in machines}
        machine_finish_time = {machine: 0 for machine in machines}
        ids = sorted(range(len(resource_requirement)), key = lambda i: np.min(list(timing[i].values())), reverse= True) 
        resource_occupancy = {machine: [0]*num_resources for machine in machines}
        
        def put(machine, job_id):  
            machine_resource_constraint = resource_costraint[machine] 
            commit_time = machine_commit_time[machine]
            job_time = timing[job_id][machine]
            finish_time = max(commit_time + job_time, machine_finish_time[machine])
            resource_usage_before = resource_used[machine]
            resource_usage_after = [x+y for x, y in zip(resource_requirement[job_id][machine], resource_usage_before)]
            need_reset = reduce(lambda x,y: x or y, [x>y for x,y in zip(resource_usage_after, machine_resource_constraint)], False)
            if need_reset:
                resource_usage_before = [0] * num_resources 
                resource_usage_after = resource_requirement[job_id][machine]
                commit_time = machine_finish_time[machine] 
                finish_time = commit_time + job_time 
            return finish_time, commit_time, job_time, resource_usage_before, resource_usage_after  
        
        placement = {}
        schedule = {}
                
        for id in ids:
            best_finish_time = math.inf 
            best_machine = None
            for machine in timing[id].keys():
                finish_time, commit_time,job_time,_,_ = put(machine, id)
                if finish_time < best_finish_time: 
                    best_finish_time = finish_time
                    best_machine = machine 
            assert best_finish_time < math.inf 
            finish_time, commit_time, job_time, resource_usage_before, resource_usage_after = put(best_machine, id)
            placement[id] = (best_machine, resource_usage_before)
            schedule[id] = commit_time
            resource_used[best_machine] = resource_usage_after 
            machine_finish_time[best_machine] = finish_time 
            machine_commit_time[best_machine] = commit_time
            for i, r in enumerate(resource_requirement[id][best_machine]):
                resource_occupancy[best_machine][i] += job_time * r       
        
        elapsed = max(list(machine_finish_time.values()))
        
        for machine in machines: 
            for i in range(num_resources):
                if elapsed * resource_costraint[machine][i] == 0:
                    print ('resource constraint', resource_costraint)
                    print('timing', timing)
                    print ('resource requirement', resource_requirement)
                    print ('elapsed: ', elapsed, 'machine', machine, 'i', i)
                resource_occupancy[machine][i] /= elapsed * resource_costraint[machine][i]
        # print (f'schedule {len(timing)} to {len(machines)} machines, resource_occypancy is {resource_occupancy}')
        # # print('elapsed', elapsed)
        # print('timing', timing)
        # print('resource usage', resource_requirement)
        # print('resource constraint', resource_costraint)
        
        # print('placement', placement)
        # print('schedule', schedule)
        # print('resource_occupancy', resource_occupancy)
        
        return elapsed, placement, schedule, resource_occupancy 


'''
Driver class for mapping. 
'''
class ComputationGraph:
    def __init__(self, graph, mapper, verbose = False):
        self.g = graph
        self.mapper = mapper
        self.mapper.set_cg(self) 
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
            acc = u['acc']
            c = acc2color[u['acc']]
            s += f'node{id} [label=\"{op_type}{acc}\", color=\"#{c}\"]\n'
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

    def map(self, soc: SoCBase) -> float:
        self.reset()
        return self.mapper(soc)
        # assert (self.check())

    def lower_bound(self, soc: SoCBase):
        exec_time = {}
        accs = soc.get_all_accs()
        
        for nid in nx.topological_sort(self.g):
            node = self.g.nodes[nid]
            task = node['task']
            acc_name = accs[MapperBase.op2task[node['op'].name]][0]
            acc = soc.accelerator_graph.nodes[acc_name]['acc']
            params = task.get_params()
            exec_time[nid] = acc.evaluate_compute(*params)[0] + (max(exec_time[pred] for pred in self.g.pred[nid]) if self.g.pred[nid] else 0)
        return max(exec_time.values())

    def visualize_packing(self, soc: SoCBase, complete_time: float, filepath: str = "packing"):
        acc2peoffset = {}
        acc2streamoffset = {}
        pe_offset = 0
        num_streams = 0
        for _, acc in soc.accelerator_graph.nodes.data('acc'):
            acc2peoffset[acc.name] = pe_offset 
            pe_offset += acc.num_pes
            acc2streamoffset[acc.name] = num_streams 
            num_streams+= acc.num_streams()
            
        complete_time /= 100 
        pe_offset /= 100
        img = Image.new("RGB", (int(complete_time), int(pe_offset)))

        for _, task in self.g.nodes.data('task'):
            img1 = ImageDraw.Draw(img)
            # img1.rectangle([(task.compute_start / 100, (acc2peoffset[task.acc] + task.pe_start) / 100), (task.compute_finish / 100, (acc2peoffset[task.acc] + task.pe_finish) / 100)], fill ="yellow", outline ="red")
            img1.rectangle([((acc2streamoffset[task.acc] + task.stream) * 10, task.compute_start / 100), ((acc2streamoffset[task.acc] + task.stream + 1) * 10, task.compute_finish / 100)], fill ="yellow", outline ="red")
        for v in acc2peoffset.values():
            img1 = ImageDraw.Draw(img)
            img1.line([(0, v), (complete_time, v)], fill = 'red', width = 0)
        img.save(f'{filepath}.png')
        
            
class MapperBase:
    op2task = {
        Op.OpName.ConvOp.Conv2d: "Conv2d", 
        Op.OpName.ConvOp.DepthwiseConv2d: "Depthwise", 
        Op.OpName.MatrixOp.Gemm:"Gemm",
        Op.OpName.MatrixOp.MatMul:"Gemm"
    }
    # the complete time if we map op to acc 
    def __init__(self, verbose: bool = False):
        self.g = nx.DiGraph()
        self.verbose = verbose 
    
    def set_cg(self, cg:ComputationGraph):
        self.cg = cg 
    
    '''
    Commit a bunch of ops. 
    '''
    def commit(self, soc: SoCBase, nodes: List[int], streams: Tuple[Tuple[AcceleratorBase, int]], simulate: bool = False):
        current_time = soc.elapsed_time
        assert len(nodes) == len(streams)
        for id, stream in zip(nodes, streams):
            soc.push_task(self.cg.g.nodes[id]['task'], *stream)
        
        complete_time = soc.commit_all_tasks() 
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
    # round robin
    def commit_rr(self, soc: SoCBase, nodes: List[int], simulate: bool = False):
        current_time = soc.elapsed_time
        streams = soc.get_all_streams()
        
        counter = {task_kind: 0 for task_kind in streams}
        
        for id in nodes:
            task = self.cg.g.nodes[id]['task']
            idx = counter[task.task_kind]
            soc.push_task(self.cg.g.nodes[id]['task'], *streams[task.task_kind][idx])
            counter[task.task_kind] = (idx+1) % len(streams[task.task_kind])
        
        complete_time = soc.commit_all_tasks() 
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
    
    def get_compute_time(self, accs):
        compute_times = {}
        for nid, op in self.cg.g.nodes.data('op'):
            avail_accs = accs[MapperBase.op2task[op.name]]
            compute_times[nid] = {acc: self.soc.eval_computation(self.cg.g.nodes[nid]['task'], acc) 
                                  for acc in avail_accs}
        return compute_times 

    def get_resource_usage(self, accs):
        resource_usages = {}
        for nid, op in self.cg.g.nodes.data('op'):
            avail_accs = accs[MapperBase.op2task[op.name]]
            resource_usages[nid] = {acc: (1, self.soc.eval_pe_usage(self.cg.g.nodes[nid]['task'], acc)) 
                                  for acc in avail_accs}
        return resource_usages 
    
    '''
    The mapping implementation. The mapper should annotate the start/end/acc attributes for each node.
    '''
    def __call__(self, soc: SoCBase):
        raise NotImplementedError()

'''
A converter to transform graph ir into networkx graph
'''
class GraphIRConverter(GraphVisitor):
    def __init__(self):
        self.op2index = {}
        self.g = nx.DiGraph()
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
        self.visit_graph(graph, specify_subgraphs=specify_subgraphs, init_state=init_state)
        return self.g 
    
    def postprocess(self):
        self.g = nx.transitive_closure(self.g)
        considered_ops = [Op.OpName.ConvOp.Conv2d, Op.OpName.ConvOp.DepthwiseConv2d, Op.OpName.MatrixOp.Gemm, Op.OpName.MatrixOp.MatMul]
        self.g = self.g.subgraph([id for id in self.g.nodes if self.g.nodes[id]['op'].name in considered_ops])
        G = nx.transitive_reduction(self.g)
        G.add_nodes_from(self.g.nodes(data = True))
        self.g = G
        
        for id in self.g.nodes:
            node = self.g.nodes[id]
            task = node['task']
            task.name = f'T{id}'
            task.depend_tasks = [self.g.nodes[i]['task'] for i in self.g.pred[id]]
            task.params = node['op'].get_config()
            if node['op'].name == Op.OpName.ConvOp.Conv2d:
                task.task_kind = "Conv2d"
            elif node['op'].name == Op.OpName.ConvOp.DepthwiseConv2d:
                task.task_kind = "Depthwise" 
            elif node['op'].name == Op.OpName.MatrixOp.Gemm or node['op'].name == Op.OpName.MatrixOp.MatMul:
                task.task_kind = "Gemm"
            else:
                raise RuntimeError()
        return self.g

def visualize(graph, name = 'g'):
    s = 'digraph G{\n'
    for u,op in graph.nodes.data('op'):
        s += f'node{u} [label=\"{u} {op.name}\"]\n'
    for u,v in graph.edges:
        s += f'node{u} -> node{v}\n'
    s += '}\n'
    
    with open('tmp.gv', 'w') as f:
        f.write(s)
    
    cmd = f'dot -Tpng tmp.gv -o {name}.png'
    os.system(cmd)

def visualize_basic(graph, name = 'g'):
    s = 'digraph G{\n'
    node2name = lambda x: f'S{-x}' if x < 0 else f'N{x}'
    for u,v in graph.edges:
        s += f'{node2name(u)} -> {node2name(v)}\n'
    s += '}\n'
    
    with open('tmp.gv', 'w') as f:
        f.write(s)
    
    cmd = f'dot -Tpng tmp.gv -o {name}.png'
    os.system(cmd)
    
def visualize_serialize(graph, name = 'g'):
    s = 'digraph G{\n'
    node2name = {x:i+1 for i,x in enumerate(graph.nodes)}
    for u,v in graph.edges:
        s += f'{node2name[u]} -> {node2name[v]}\n'
    s += '}\n'
    
    with open('tmp.gv', 'w') as f:
        f.write(s)
    
    cmd = f'dot -Tpng tmp.gv -o {name}.png'
    os.system(cmd)
    
def visualize_subgraph(graph, subgraphs: List[List[int]], name = 'g'):
    s = 'digraph G{\n'
    s += "\tnode [style=filled]\n"
    node2name = lambda x: f'S{-x}' if x < 0 else f'N{x}'
    for subgraph in subgraphs:
        c = f'{hex(random.randint(0,256*256*256))}'
        for u in subgraph:
            s += f'{node2name(u)} [color=\"#{c}\"]\n'
    for u,v in graph.edges:
        s += f'{node2name(u)} -> {node2name(v)}\n'
    s += '}\n'
    
    with open('tmp.gv', 'w') as f:
        f.write(s)
    
    cmd = f'dot -Tpng tmp.gv -o {name}.png'
    os.system(cmd)

def get_graph(models: List[str]):
    irConverter = GraphIRConverter()
    n_node = 0
    for model in models:
        if model == 'resnet18':
            model_path = "./graph/raw_resnet18.onnx"
        elif model == "mobilenet":
            model_path = "./graph/raw_mobilenetv2.onnx"
        elif model == "resnet50":
            model_path = "./graph/raw_resnet50.onnx"
        elif model == "yolo": 
            model_path = "./graph/yolov5s_640x640.simplify.onnx"
        elif model == "GoogLeNet":
            model_path = "./graph/googlenet-12.onnx"
        elif model == "SSD-M":
            model_path = './graph/ssd_mobilenet_v1_10.onnx'
        elif model == 'efficientnet':
            model_path = './graph/efficientnet-lite4-11.onnx'
        elif model == "super_resolution":
            model_path = "./graph/super_resolution.onnx"
        elif model == 'bert':
            model_path = "./graph/simplified_bert_base.onnx"
        elif model == "lstm":
            model_path = "./graph/multi_layer_lstm.onnx"
        else:
            raise RuntimeError(f'unknown model {model}')
        convertor = ONNXConvertor(model_path, inference=True)
        graph  = convertor.parse()
        irConverter(graph)
    
    return irConverter.postprocess()
