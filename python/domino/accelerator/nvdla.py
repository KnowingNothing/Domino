import os
import copy
from typing import Dict, Any
from collections import OrderedDict
from ..base import AcceleratorBase, AccTask
from ..utils import run_maestro, generate_maestro_command, find_maestro
from .conv_acc import ConvAccelerator
from .gemm_acc import GemmAccelerator


class NVDLA(ConvAccelerator):
    def __init__(self, name, n_stream=1, freq=200, num_pes=128*128, noc_bw=81920000, off_chip_bw=81920000, l1_size=4000000, l2_size=24000000) -> None:
        super(NVDLA, self).__init__(name, n_stream, freq=freq, num_pes=num_pes, noc_bw=noc_bw,
                                    off_chip_bw=off_chip_bw, l1_size=l1_size, l2_size=l2_size)

    def get_mapping(self, H, W, P, Q, K, C, R, S, stride_h, stride_w):
        mapping = ("Network sample_net {\n"
                   "Layer Conv2d {\n"
                   "Type: CONV\n"
                   "Stride { "
                   f"X: {stride_h}, Y: {stride_w} "
                   "}\n"
                   "Dimensions { "
                   f"K: {K}, C: {C}, R: {R}, S: {S}, Y: {H}, X: {W} "
                   "}\n"
                   "Dataflow {\n"
                   "        SpatialMap(1,1) K;\n"
                   "        TemporalMap(64,64) C;\n"
                   "        TemporalMap(Sz(R),Sz(R)) R;\n"
                   "        TemporalMap(Sz(S),Sz(S)) S;\n"
                   "        TemporalMap(Sz(R),1) Y;\n"
                   "        TemporalMap(Sz(S),1) X;\n"
                   "        Cluster(64, P);\n"
                   "        SpatialMap(1,1) C;\n"
                   "        TemporalMap(Sz(R),1) Y;\n"
                   "        TemporalMap(Sz(S),1) X;\n"
                   "        TemporalMap(Sz(R),Sz(R)) R;\n"
                   "        TemporalMap(Sz(S),Sz(S)) S;\n"
                   "}\n"
                   "}\n"
                   "}\n")

        return mapping

    def spatial_used_pes(self, H, W, P, Q, K, C, R, S, stride_h, stride_w):
        """
        Return how many PEs are actually needed
        This is calculated according to the mapping
        """
        return min(K * 64, self.num_pes)

    def __str__(self) -> str:
        return f'NVDLA{self.topo_id}'

    def __repr__(self) -> str:
        return f'NVDLA{self.topo_id}'


class GemmNVDLA(GemmAccelerator):
    def __init__(self, name, n_stream=1, freq=200, num_pes=128*128, noc_bw=81920000, off_chip_bw=81920000, l1_size=4000000, l2_size=24000000) -> None:
        super(GemmNVDLA, self).__init__(name, n_stream, freq=freq, num_pes=num_pes, noc_bw=noc_bw,
                                        off_chip_bw=off_chip_bw, l1_size=l1_size, l2_size=l2_size)

    def get_mapping(self, M, N, K):
        # N * 64
        mapping = ("Network sample_net {\n"
                   "Layer GEMM {\n"
                   "Type: GEMM\n"
                   "Dimensions { "
                   f"K: {K}, M: {M}, N: {N} "
                   "}\n"
                   "Dataflow {\n"
                   "        SpatialMap(1, 1) N;\n"
                   "        TemporalMap(64, 64) K;\n"
                   "        TemporalMap(1, 1) M;\n"
                   "        Cluster(64, P);\n"
                   "        SpatialMap(1, 1) K\n"
                   "        TemporalMap(1, 1) M;\n"
                   "        TemporalMap(1, 1) N;\n"
                   "}\n"
                   "}\n"
                   "}\n")

        return mapping

    def spatial_used_pes(self, M, N, K):
        """
        Return how many PEs are actually needed
        This is calculated according to the mapping
        """
        return min(N * 64, self.num_pes)

    def __str__(self) -> str:
        return f'GemmNVDLA{self.topo_id}'

    def __repr__(self) -> str:
        return f'GemmNVDLA{self.topo_id}'
