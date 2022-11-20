import os
import copy
import pickle as pkl
import networkx as nx
from typing import Dict, Any, List, Tuple
from collections import OrderedDict
import numpy as np
from .. import global_timer


class AccTask(object):
    unique_id = 0

    def __init__(self, name, task_kind=None, params: Dict[str, Any] = {}, depend_tasks: List["AccTask"] = []) -> None:
        assert task_kind in [None, "Conv2d", "Depthwise", "Gemm"]
        self.name = name  # The unique id in compute_graph
        self.unique_id = AccTask.unique_id
        AccTask.unique_id += 1
        self.task_kind = task_kind
        self.params = params
        self.depend_tasks = depend_tasks  # the parent tasks of this task
        # for visualization
        self.compute_start = 0
        self.compute_finish = 0
        self.pe_start = 0
        self.pe_finish = 0
        self.acc = None
        self.stream = 0

    def get_params(self):
        if self.task_kind == "Conv2d":
            H, W, P, Q, K, C, R, S = [self.params[x] for x in "HWPQKCRS"]
            stride_h = self.params["stride_h"]
            stride_w = self.params["stride_w"]
            return (H, W, P, Q, K, C, R, S, stride_h, stride_w)
        elif self.task_kind == 'Gemm':
            B, M, N, K = [self.params[x] for x in 'BMNK']
            return (B, M, N, K)
        elif self.task_kind == 'Depthwise':
            H, W, P, Q, K, M, R, S = [self.params[x] for x in 'HWPQKMRS']
            stride_h = self.params['stride_h']
            stride_w = self.params['stride_w']
            return (H, W, P, Q, K, M, R, S, stride_h, stride_w)
        else:
            raise NotImplementedError()

    def get_output_data_volume(self):
        """
        The unit is byte.
        Currently, we only consider int8 precision
        """
        if self.task_kind == "Conv2d":
            H, W, P, Q, K, C, R, S = [self.params[x] for x in "HWPQKCRS"]
            return P * Q * K
        elif self.task_kind == 'Gemm':
            B, M, N = [self.params[x] for x in 'BMN']
            return B*M*N
        elif self.task_kind == 'Depthwise':
            K, P, Q = [self.params[x] for x in 'KPQ']
            return K*P*Q
        else:
            raise NotImplementedError()

    def __hash__(self) -> int:
        return hash((self.name, self.task_kind, tuple(self.params)))

    def __str__(self) -> str:
        return f"Task({self.name}, {self.task_kind}, {self.params})"

    def __repr__(self) -> str:
        return str(self)

# communication time 

class AccStream(object):
    def __init__(self, idx) -> None:
        self.idx = idx
        self._stream = []
        self._to_commit = []
        self._elapsed_time = 0
        self.logs = []

    def __getitem__(self, idx):
        return self._stream[idx]

    def get_current_elapsed_time(self):
        return self._elapsed_time

    def num_tasks(self):
        return len(self._stream)

    def add(self, task: AccTask):
        assert isinstance(task, AccTask)
        self._stream.append(task)

    def retire(self, task, comm_time, compute_time):
        elapsed_time = comm_time + compute_time 
        assert len(self._to_commit) > 0
        assert task == self._to_commit[-1], f"stream: {self.idx}: {task} vs {self._to_commit[-1]}"
        self.logs.append([task, self._elapsed_time, {'elapsed': elapsed_time, 'compute': compute_time, 'communication': comm_time}])
        self._to_commit = self._to_commit[:-1]
        self._elapsed_time += elapsed_time
        return self._elapsed_time

    def find(self, task):
        """
        Return -1 if not found
        else, return the task position in current stream
        """
        idx = -1
        for i, t in enumerate(self._stream):
            if task == t:
                idx = i
                break
        return idx

    def empty(self):
        return len(self._stream) == 0

    def head(self):
        return self._stream[0]

    def prepare_to_commit(self):
        self._to_commit.append(self.head())
        self._stream = self._stream[1:]

    def report(self):
        occupation = sum([x[2]['elapsed'] for x in self.logs]) / self._elapsed_time 
        comm_rate = sum([x[2]['communication'] for x in self.logs]) / self._elapsed_time 
        compute_portion = sum([x[2]['compute'] for x in self.logs]) / self._elapsed_time 
        print(
            f"\tstream {self.idx}: {len(self.logs)} task, {occupation} occupied, {comm_rate} communication, {compute_portion} computation")
    
    def profile(self):
        compute_time = sum(x[2]['compute'] for x in self.logs)
        comm_time = sum(x[2]['communication'] for x in self.logs)
        idle_time = self._elapsed_time - compute_time - comm_time 
        return {'compute': compute_time, 'communication': comm_time, 'idle': idle_time}

class AcceleratorBase(object):

    def __init__(self, name, n_stream, supported_task, freq=200, num_pes=65536, noc_bw=81920000, off_chip_bw=81920000, l1_size=4000000, l2_size=24000000) -> None:
        print(f"create {name} with {n_stream} streams")
        self.name = name
        self.supported_task = supported_task
        self.freq = freq * 1e6 # Hz
        self.num_pes = num_pes
        self.noc_bw = noc_bw  # byte/cycle
        self.off_chip_bw = off_chip_bw  # byte/s
        self.l1_size = l1_size  # byte
        self.l2_size = l2_size  # byte
        self.unique_stream_id = 0  # increase only
        self.streams = OrderedDict()
        self.streams[self.unique_stream_id] = AccStream(self.unique_stream_id)
        self.board = {}  # record mapping task to stream idx
        self.topo_id = (0)
        for _ in range(n_stream - 1):
            self.add_new_stream()
        self.total_energy = 0  # nJ
        
        # for visualization
        self.pe_usages = []

    def add_new_stream(self):
        self.unique_stream_id += 1
        idx = self.unique_stream_id
        self.streams[idx] = AccStream(idx)
        return idx

    def delete_stream(self, idx):
        assert idx in self.streams
        del self.streams[idx]

    def num_streams(self):
        return len(self.streams)

    def get_stream(self, idx):
        assert idx in self.streams
        return self.streams[idx]

    def __getitem__(self, idx):
        assert idx in self.streams
        return self.streams[idx]

    def push_task_to_stream(self, idx, task: AccTask):
        stream = self.get_stream(idx)
        stream.add(task)
        self.board[task] = idx

    def snapshot(self):
        ret = copy.deepcopy(self)
        return ret

    def get_mapping(self, *args):
        raise NotImplementedError()

    def spatial_used_pes(self, *args):
        raise NotImplementedError()

    def evaluate_compute(self, *args):
        """
        Calculate compute runtime seconds and energy (nJ)
        """
        raise NotImplementedError()

    def evaluate_fetch_data(self, task: AccTask, soc: "SoCBase"):
        """
        Calculate fetch data runtime seconds
        """
        # max_transfer_time = 0
        sum_transfer_time = 0
        for ptask in task.depend_tasks:
            transfer_time = soc.evaluate_data_transfer(ptask, task)
            sum_transfer_time += transfer_time
            # max_transfer_time = max(max_transfer_time, transfer_time)
        return sum_transfer_time

    def commit_current_tasks(self, soc: "SoCBase"):
        phase = {}
        cur_pe_usage = 0
        total_tasks = 0
        for idx, stream in self.streams.items():
            total_tasks += stream.num_tasks()

        def commit():
            nonlocal phase
            nonlocal cur_pe_usage
            sync_time = 0
            old_time = max(self.get_stream(idx)._elapsed_time for idx in range(self.num_streams()))

            pe_offset = 0
            pe_amount = 0
            comm_amount = 0
            max_power = 0
            for idx, task in phase.items():
                stream = self.get_stream(idx)
                task_params = task.get_params()
                fetch_data_cost = self.evaluate_fetch_data(task, soc)
                compute_time_cost, power = self.evaluate_compute(
                    *task_params)
                max_power = max(max_power, power)
                sync_time = max(sync_time, stream.retire(
                    task, fetch_data_cost, compute_time_cost))

                task.pe_start = pe_offset
                pe_usage = self.spatial_used_pes(*task.get_params())
                pe_offset += pe_usage 
                pe_amount += pe_usage * compute_time_cost
                comm_amount += pe_usage * fetch_data_cost
                task.pe_finish = pe_offset
                task.compute_start = stream._elapsed_time
                task.compute_finish = task.compute_start + fetch_data_cost + compute_time_cost
                task.acc = self.name
                task.stream = idx
            phase_time = sync_time - old_time
            self.pe_usages.append({'occupancy': pe_offset / self.num_pes, 
                                   'amount': pe_amount, 
                                   'comm_amount': comm_amount, 
                                   'phase_amount': phase_time * self.num_pes})
            self.total_energy += phase_time * max_power
            for idx in range(self.num_streams()):
                self.get_stream(idx)._elapsed_time = sync_time
            phase = {}
            cur_pe_usage = 0

        while total_tasks > 0:
            # this traverse is ordered according to idx because we use OrderedDict
            pushed = False
            for idx, stream in self.streams.items():
                if not stream.empty():
                    task = stream.head()
                    task_params = task.get_params()
                    task_pe_usage = self.spatial_used_pes(*task_params)
                    assert cur_pe_usage <= self.num_pes, f"{cur_pe_usage} vs {self.num_pes}"
                    if cur_pe_usage + task_pe_usage > self.num_pes:
                        total_tasks -= len(phase)
                        commit()
                    if idx in phase:
                        # note that for every traverse of the head
                        # we need to clear the phase to guarantee
                        # correct dependency within one stream and
                        # tasks within one phase are purely independent
                        continue
                    cur_pe_usage += task_pe_usage
                    phase[idx] = task
                    stream.prepare_to_commit()

                    pushed = True
            if not pushed and len(phase):
                total_tasks -= len(phase)
                commit()
        assert len(phase) == 0, f"{phase}"
        return max([x._elapsed_time for x in self.streams.values()])

    def sync(self, time):
        for stream in self.streams.values():
            assert time >= stream._elapsed_time
            stream._elapsed_time = time

    def get_current_elapsed_time(self):
        return max([stream.get_current_elapsed_time() for stream in self.streams.values()])

    def get_current_energy_consumption(self):
        return self.total_energy

    compute_cache = {}

    @staticmethod
    def load_cache(dir: str = './.cache'):
        file = os.path.join(dir, "accelerator.pkl")
        if os.path.exists(file):
            with open(file, 'rb') as f:
                AcceleratorBase.compute_cache = pkl.load(f)

    @staticmethod
    def store_cache(dir: str = './.cache/'):
        file = os.path.join(dir, "accelerator.pkl")
        if os.path.exists(file):
            with open(file, 'wb') as f:
                pkl.dump(AcceleratorBase.compute_cache, f)

    def report(self):
        print(f"{self.name}: ")
        print (f'\tEnergy consumption {self.get_current_energy_consumption()}')
        for stream_id in range(self.num_streams()):
            self.get_stream(stream_id).report()

    def profile(self):
        data = {'compute': 0, 'communication': 0, 'idle': 0}
        for idx in range(self.num_streams()):
            stream = self.get_stream(idx)
            stream_data = stream.profile()
            for k in data: 
                data[k] += stream_data[k]
        return data 

class SoCBase(object):
    def __init__(self, accelerator_graph: nx.DiGraph, name = 'SoC') -> None:
        self.accelerator_graph = accelerator_graph
        self.name = name
        self.elapsed_time = 0
        self._bind_table = {}  # task id to (accelerator_id, stream_id)

    def evaluate_data_transfer(self, task_from: AccTask, task_to: AccTask):
        assert task_from.unique_id in self._bind_table
        assert task_to.unique_id in self._bind_table
        acc_from, stream_from = self._bind_table[task_from.unique_id]
        acc_to, stream_to = self._bind_table[task_to.unique_id]
        transfer_volume = task_from.get_output_data_volume() / 1e9  # unit is GB
        try:
            # unit is GB/s
            bw = self.accelerator_graph[acc_from][acc_to]["bandwidth"]
            return transfer_volume / bw
        except Exception as e:
            print(f"Can't get the bandwidth from {acc_from} to {acc_to}")
            raise e

    def push_task(self, task, acc: str, stream_id):
        assert isinstance(acc, str)
        self._bind_table[task.unique_id] = (acc, stream_id)
        self.accelerator_graph.nodes[acc]['acc'].push_task_to_stream(
            stream_id, task)

    def get_all_streams(self) -> Dict[str, List[Tuple[str, str]]]:
        ret = {}
        for name, acc in self.accelerator_graph.nodes.data('acc'):
            for task_kind in acc.supported_task:
                if task_kind not in ret:
                    ret[task_kind] = []
                ret[task_kind] += [(name, i) for i in range(acc.num_streams())]
        return ret

    def commit_all_tasks(self):
        self.elapsed_time = 0.0
        for acc_name in self.accelerator_graph.nodes:
            acc = self.accelerator_graph.nodes[acc_name]["acc"]
            self.elapsed_time = max(
                self.elapsed_time, acc.commit_current_tasks(self))
            for i in range(acc.num_streams()):
                assert acc[i].empty()
        # global sync
        for _, acc in self.accelerator_graph.nodes.data('acc'):
            acc.sync(self.elapsed_time)

        return self.elapsed_time

    def get_current_elapsed_time(self):
        max_time = 0
        for acc_name in self.accelerator_graph.nodes:
            acc = self.accelerator_graph.nodes[acc_name]["acc"]
            max_time = max(max_time, acc.get_current_elapsed_time())
        return max_time

    def get_current_energy_consumption(self):
        total = 0
        for acc_name in self.accelerator_graph.nodes:
            acc = self.accelerator_graph.nodes[acc_name]["acc"]
            total += acc.get_current_energy_consumption()
        return total

    def snapshot(self):
        return copy.deepcopy(self)

    def eval(self,
             curr_task: Tuple[AccTask, "AcceleratorName"],
             input_tasks: List[Tuple[AccTask, "AcceleratorName"]]
             ):
        acc_name, task = curr_task
        acc = self.accelerator_graph.nodes[acc_name]['acc']
        params = task.get_params()
                     
        fetch_data_cost = max(
            task_from.get_output_data_volume() / 1e9 /
            self.accelerator_graph[acc_from][acc_name]['bandwidth']
            for acc_from, task_from in input_tasks
        ) if len(input_tasks) else 0

        compute_time_cost = acc.evaluate_compute(*params)[0]

        return fetch_data_cost + compute_time_cost, acc.spatial_used_pes(*params)

    def eval_communication(self, 
        curr_task: Tuple[AccTask, "AcceleratorName"], 
        input_tasks: List[Tuple[AccTask, "AcceleratorName"]]
    ):
        acc_name, _  = curr_task
        fetch_data_cost = sum(
            task_from.get_output_data_volume() / 1e9 / self.accelerator_graph[acc_from][acc_name]['bandwidth']
            for acc_from, task_from in input_tasks
        ) if len(input_tasks) else 0
        
        return fetch_data_cost

    def eval_computation(self, task, acc_name):
        acc = self.accelerator_graph.nodes[acc_name]['acc']
        return acc.evaluate_compute(*task.get_params())[0]
    
    def eval_pe_usage(self, task, acc_name):
        acc = self.accelerator_graph.nodes[acc_name]['acc']
        return acc.spatial_used_pes(*task.get_params())

    def get_acc2task_kinds(self):
        return {acc_name: acc.supported_task for acc_name, acc in self.accelerator_graph.nodes.data('acc')}

    def get_all_accs(self):
        ret = {}
        for name, acc in self.accelerator_graph.nodes.data('acc'):
            for task_kind in acc.supported_task:
                if task_kind not in ret:
                    ret[task_kind] = []
                ret[task_kind].append(acc.name)
        return ret

    def report(self):
        for _, acc in self.accelerator_graph.nodes.data('acc'):
            acc.report()
            
    def store_pe_curve(self, path):
        data = {}
        for name, acc in self.accelerator_graph.nodes.data('acc'):
            data[name] = {'amount': self.elapsed_time * acc.num_pes, 
                          'compute_amount': sum(x['amount'] for x in acc.pe_usages), 
                          'comm_amount': sum(x['comm_amount'] for x in acc.pe_usages),
                          'phase_amount': sum(x['phase_amount'] for x in acc.pe_usages),
                          'occupancy': np.mean([x['occupancy'] for x in acc.pe_usages])}
        with open(path, 'wb') as f:
            pkl.dump(data, f)


    def get_resource_limit(self, acc: str):
        acc = self.accelerator_graph.nodes[acc]['acc']
        return {"num_stream": acc.num_streams(), 'num_pes': acc.num_pes}
    
    def get_all_resource_limit(self):
        return {name: [acc.num_streams(), acc.num_pes] 
                for name, acc in self.accelerator_graph.nodes.data('acc')}
    
    def get_machines(self):
        return [str(x) for x in self.accelerator_graph.nodes]
    
    def profile(self):
        data = {}
        for name, acc in self.accelerator_graph.nodes.data('acc'):
            data[name] = acc.profile()
        return {'elapsed': self.elapsed_time, 'profile': data} 
