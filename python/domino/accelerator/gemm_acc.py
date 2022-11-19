import os
import copy
from typing import Dict, Any
from collections import OrderedDict
from ..base import AcceleratorBase, AccTask
from ..utils import run_maestro, generate_maestro_command, find_maestro
from .. import global_timer
import math


class GemmAccelerator(AcceleratorBase):
    def __init__(self, name, n_stream=1, freq=200, num_pes=128*128, noc_bw=81920000, off_chip_bw=81920000, l1_size=4000000, l2_size=24000000):
        super(GemmAccelerator, self).__init__(name, n_stream, ['Gemm'],
                                              freq, num_pes, noc_bw, off_chip_bw, l1_size, l2_size)

    def push_task_to_stream(self, idx, task: AccTask):
        assert task.task_kind == "Gemm"
        super(GemmAccelerator, self).push_task_to_stream(idx, task)

    def evaluate_compute(self, *args):
        key = ("gemm", args)
        if key not in AcceleratorBase.compute_cache:
            if len(args) == 3:
                M, N, K = args
                batches = 1
            elif len(args) >= 4:
                M, N, K = args[-3:]
                batches = math.prod(args[:-3], start=1)
            else:
                raise ValueError(f"Don't support Gemm with args: {args}")
            mapping_contents = self.get_mapping(M, N, K)
            mapping_file = "gemm_sample_mapping"

            if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
                os.remove(f"{mapping_file}.m")
            global_timer.start('write file')
            with open(f"{mapping_file}.m", "w") as fout:
                fout.write(mapping_contents)
            global_timer.stop('write file')

            global_timer.start('maestro')
            maestro_path = find_maestro()
            command = generate_maestro_command(
                maestro_path,
                mapping_file,
                self.noc_bw,
                self.off_chip_bw,  # off_chip_bw,
                self.num_pes,  # num_pes,
                self.l1_size,  # l1_size,
                self.l2_size,  # l2_size,
            )

            results = run_maestro(mapping_file, command)
            global_timer.stop('maestro')

            if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
                os.remove(f"{mapping_file}.m")
            AcceleratorBase.compute_cache[key] = batches * \
                results.runtime[0] / self.freq
        ret = AcceleratorBase.compute_cache[key]
        return ret
