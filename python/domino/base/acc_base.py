import os
import copy
import networkx as nx
from typing import Dict, Any, List
from collections import OrderedDict


class AccTask(object):
    def __init__(self, name, task_kind, params: Dict[str, Any], depend_tasks: List["AccTask"]) -> None:
        assert task_kind in ["Conv2d", "Depthwise", "Gemm"]
        self.name = name  # The unique id in compute_graph
        self.task_kind = task_kind
        self.params = params
        self.depend_tasks = depend_tasks  # the parent tasks of this task

    def get_params(self):
        if self.task_kind == "Conv2d":
            H, W, P, Q, K, C, R, S = [self.params[x] for x in "HWPQKCRS"]
            stride_h = self.params["stride_h"]
            stride_w = self.params["stride_w"]
            return (H, W, P, Q, K, C, R, S, stride_h, stride_w)
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
        else:
            raise NotImplementedError()

    def __hash__(self) -> int:
        return hash((self.name, self.task_kind, tuple(self.params)))


class AccStream(object):
    def __init__(self) -> None:
        self._stream = []
        self._elapsed_time = 0

    def __getitem__(self, idx):
        return self._stream[idx]

    def get_current_elapsed_time(self):
        return self._elapsed_time

    def num_tasks(self):
        return len(self._stream)

    def add(self, task: AccTask):
        assert isinstance(task, AccTask)
        self._stream.append(task)

    def retire(self, task, elapsed_time):
        assert len(self._stream) > 0
        assert task == self._stream[0]
        self._stream = self._stream[1:]
        self._elapsed_time += elapsed_time

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


class AcceleratorBase(object):
    def __init__(self, name, freq=200, num_pes=65536, noc_bw=81920000, off_chip_bw=81920000, l1_size=4000000, l2_size=24000000) -> None:
        self.name = name
        self.freq = freq  # Hz
        self.num_pes = num_pes
        self.noc_bw = noc_bw  # byte/cycle
        self.off_chip_bw = off_chip_bw  # byte/s
        self.l1_size = l1_size  # byte
        self.l2_size = l2_size  # byte
        self.unique_stream_id = 0  # increase only
        self.streams = OrderedDict()
        self.streams[self.unique_stream_id] = AccStream()
        self.board = {}  # record mapping task to stream idx

    def add_new_stream(self):
        self.unique_stream_id += 1
        idx = self.unique_stream_id
        self.streams[idx] = AccStream()
        return idx

    def delete_stream(self, idx):
        assert idx in self.streams
        del self.streams[idx]

    def num_streams(self):
        return len(self.streams)

    def get_stream(self, idx):
        assert idx in self.streams
        return self.streams[idx]

    def push_task_to_stream(self, idx, task: AccTask):
        assert task.task_kind == "Conv2d"
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
        Calculate compute runtime seconds
        """
        raise NotImplementedError()

    def evaluate_fetch_data(self, task: AccTask, soc: "SoCBase"):
        """
        Calculate fetch data runtime seconds
        """
        max_transfer_time = 0
        for ptask in task.depend_tasks:
            transfer_time = soc.evaluate_data_transfer(ptask, task)
            max_transfer_time = max(max_transfer_time, transfer_time)
        return max_transfer_time

    def commit_current_tasks(self, soc: "SoCBase"):
        phase = {}
        cur_pe_usage = 0
        total_tasks = 0
        for idx, stream in self.streams.items():
            total_tasks += stream.num_tasks()

        def commit():
            nonlocal phase
            nonlocal cur_pe_usage
            for idx, task in phase.items():
                stream = self.get_stream(idx)
                task_params = task.get_params()
                fetch_data_cost = self.evaluate_fetch_data(task, soc)
                compute_time_cost = self.evaluate_compute(*task_params)
                stream.retire(task, fetch_data_cost + compute_time_cost)
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
                    if cur_pe_usage + task_pe_usage > self.num_pes:
                        commit()
                    if idx in phase:
                        # note that for every traverse of the head
                        # we need to clear the phase to guarantee
                        # correct dependency within one stream and
                        # tasks within one phase are purely independent
                        continue
                    cur_pe_usage += task_pe_usage
                    phase[idx] = task
                    total_tasks -= 1
                    pushed = True
            if not pushed and len(phase):
                commit()

    def get_current_elapsed_time(self):
        return max([stream.get_current_elapsed_time() for stream in self.streams.values()])


class SoCBase(object):
    def __init__(self, accelerator_graph: nx.DiGraph) -> None:
        self.accelerator_graph = accelerator_graph
        self._bind_table = {}  # task id to (accelerator_id, stream_id)

    def evaluate_data_transfer(self, task_from: AccTask, task_to: AccTask):
        assert task_from in self._bind_table
        assert task_to in self._bind_table
        acc_from, stream_from = self._bind_table[task_from]
        acc_to, stream_to = self._bind_table[task_to]
        transfer_volume = task_from.get_output_data_volume() / 1e9  # unit is GB
        try:
            # unit is GB/s
            bw = self.accelerator_graph[acc_from][acc_to]["bandwidth"]
            return transfer_volume / bw
        except Exception as e:
            print(f"Can't get the bandwidth from {acc_from} to {acc_to}")
            raise e

    def push_task(self, task, acc: AcceleratorBase, stream_id):
        assert isinstance(acc, AcceleratorBase)
        acc.push_task_to_stream(stream_id, task)

    def commit_all_tasks(self):
        for acc_name in self.accelerator_graph.nodes:
            acc = self.accelerator_graph.nodes[acc_name]["acc"]
            acc.commit_current_tasks(self)

    def get_current_elapsed_time(self):
        max_time = 0
        for acc_name in self.accelerator_graph.nodes:
            acc = self.accelerator_graph.nodes[acc_name]["acc"]
            max_time = max(max_time, acc.get_current_elapsed_time())
        return max_time
