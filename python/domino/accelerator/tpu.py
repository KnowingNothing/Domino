import os
import copy
from typing import Dict, Any
from collections import OrderedDict
from ..base import MaestroAcceleratorBase, AccTask
from ..utils import run_maestro, generate_maestro_command, find_maestro
from .gemm_acc import MaestroGemmAccelerator
from .conv_acc import MaestroConvAccelerator


class MaestroGemmTPU(MaestroGemmAccelerator):
    def __init__(self, name, n_stream=1, freq=200, num_pes=128*128, noc_bw=81920000, off_chip_bw=81920000, l1_size=4000000, l2_size=24000000) -> None:
        super(MaestroGemmTPU, self).__init__(name, n_stream, freq=freq, num_pes=num_pes, noc_bw=noc_bw,
                                      off_chip_bw=off_chip_bw, l1_size=l1_size, l2_size=l2_size)

    def get_mapping(self, M, N, K):
        # Mx128
        mapping = ("Network sample_net {\n"
                   "Layer GEMM {\n"
                   "Type: GEMM\n"
                   "Dimensions { "
                   f"K: {K}, M: {M}, N: {N} "
                   "}\n"
                   "Dataflow {\n"
                   "        SpatialMap(1, 1) M;\n"
                   "        TemporalMap(128, 128) N;\n"
                   "        TemporalMap(16, 16) K;\n"
                   "        Cluster(128, P);\n"
                   "        SpatialMap(1, 1) N;\n"
                   "        TemporalMap(1, 1) M;\n"
                   "        TemporalMap(16, 16) K;\n"
                   "}\n"
                   "}\n"
                   "}\n")

        return mapping

    def spatial_used_pes(self, B, M, N, K):
        """
        Return how many PEs are actually needed
        This is calculated according to the mapping
        """
        return min(M * 128, self.num_pes)

    def __str__(self) -> str:
        return f'MaestroGemmTPU{self.topo_id}'

    def __repr__(self) -> str:
        return f'MaestroGemmTPU{self.topo_id}'


class MaestroConvTPU(MaestroConvAccelerator):
    def __init__(self, name, n_stream=1, freq=200, num_pes=128*128, noc_bw=81920000, off_chip_bw=81920000, l1_size=4000000, l2_size=24000000) -> None:
        super(MaestroConvTPU, self).__init__(name, n_stream, freq=freq, num_pes=num_pes, noc_bw=noc_bw,
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
                   "        TemporalMap(128,128) C;\n"
                   "        TemporalMap(Sz(R),Sz(R)) R;\n"
                   "        TemporalMap(Sz(S),Sz(S)) S;\n"
                   "        TemporalMap(Sz(R),1) Y;\n"
                   "        TemporalMap(Sz(S),1) X;\n"
                   "        Cluster(128, P);\n"
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
        return min(K * 128, self.num_pes)

    def __str__(self) -> str:
        return f'MaestroConvTPU{self.topo_id}'

    def __repr__(self) -> str:
        return f'MaestroConvTPU{self.topo_id}'
