import os
import copy
from typing import Dict, Any
from collections import OrderedDict
from ..base import MaestroAcceleratorBase, AccTask
from ..utils import run_maestro, generate_maestro_command, find_maestro
from .conv_acc import MaestroConvAccelerator
from .depthwise_acc import MaestroDepthwiseAccelerator


class MaestroDepthwiseShiDianNao(MaestroDepthwiseAccelerator):
    def __init__(self, name, n_stream=1, freq=200, num_pes=128*128, noc_bw=81920000, off_chip_bw=81920000, l1_size=4000000, l2_size=24000000) -> None:
        super(MaestroDepthwiseShiDianNao, self).__init__(name, n_stream, freq=freq, num_pes=num_pes, noc_bw=noc_bw,
                                                  off_chip_bw=off_chip_bw, l1_size=l1_size, l2_size=l2_size)

    def get_mapping(self, H, W, P, Q, K, M, R, S, stride_h, stride_w):
        mapping = ("Network sample_net {\n"
                   "Layer DepthwiseConv2d {\n"
                   "Type: DSCONV\n"
                   "Stride { "
                   f"X: {stride_h}, Y: {stride_w} "
                   "}\n"
                   "Dimensions { "
                   f"K: {M}, C: {K}, R: {R}, S: {S}, Y: {H}, X: {W} "
                   "}\n"
                   "Dataflow {\n"
                   "        TemporalMap(1,1) C;\n"
                   "        SpatialMap(Sz(R), 1) Y;\n"
                   "        TemporalMap(64,64) X;\n"
                   "        TemporalMap(Sz(R), Sz(R)) R;\n"
                   "        TemporalMap(Sz(S), Sz(S)) S;\n"
                   "        Cluster(64, P);\n"
                   "        SpatialMap(Sz(S), 1) X;\n"
                   "}\n"
                   "}\n"
                   "}\n")

        return mapping

    def spatial_used_pes(self, H, W, P, Q, K, M, R, S, stride_h, stride_w):
        """
        Return how many PEs are actually needed
        This is calculated according to the mapping
        """
        return min(H * 64, self.num_pes)

    def __str__(self) -> str:
        return f'MaestroDepthwiseShiDianNao{self.topo_id}'

    def __repr__(self) -> str:
        return f'MaestroDepthwiseShiDianNao{self.topo_id}'


class MaestroConvShiDianNao(MaestroConvAccelerator):
    def __init__(self, name, n_stream=1, freq=200, num_pes=128*128, noc_bw=81920000, off_chip_bw=81920000, l1_size=4000000, l2_size=24000000) -> None:
        super(MaestroConvShiDianNao, self).__init__(name, n_stream, freq=freq, num_pes=num_pes, noc_bw=noc_bw,
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
                   "        TemporalMap(1,1) K;\n"
                   "        TemporalMap(1,1) C;\n"
                   "        SpatialMap(Sz(R), 1) Y;\n"
                   "        TemporalMap(64,64) X;\n"
                   "        TemporalMap(Sz(R), Sz(R)) R;\n"
                   "        TemporalMap(Sz(S), Sz(S)) S;\n"
                   "        Cluster(64, P);\n"
                   "        SpatialMap(Sz(S), 1) X;\n"
                   "}\n"
                   "}\n"
                   "}\n")

        return mapping

    def spatial_used_pes(self, H, W, P, Q, K, C, R, S, stride_h, stride_w):
        """
        Return how many PEs are actually needed
        This is calculated according to the mapping
        """
        return min(H * 64, self.num_pes)

    def __str__(self) -> str:
        return f'MaestroConvShiDianNao{self.topo_id}'

    def __repr__(self) -> str:
        return f'MaestroConvShiDianNao{self.topo_id}'
