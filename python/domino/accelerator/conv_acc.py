import os
import copy
from typing import Dict, Any
from collections import OrderedDict
from ..base import AcceleratorBase, AccTask
from ..utils import run_maestro, generate_maestro_command, find_maestro


class ConvAccelerator(AcceleratorBase):

    def push_task_to_stream(self, idx, task: AccTask):
        assert task.task_kind == "Conv2d"
        super(ConvAccelerator, self).push_task_to_stream(idx, task)

    def evaluate_compute(self, H, W, P, Q, K, C, R, S, stride_h, stride_w):
        mapping_contents = self.get_mapping(
            H, W, P, Q, K, C, R, S, stride_h, stride_w)
        mapping_file = "sample_mapping"

        if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
            os.remove(f"{mapping_file}.m")

        with open(f"{mapping_file}.m", "w") as fout:
            fout.write(mapping_contents)

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

        if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
            os.remove(f"{mapping_file}.m")

        return results.runtime[0] / self.freq