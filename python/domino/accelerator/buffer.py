from ..base import HardwareLevel
from typing import Optional


class Buffer(HardwareLevel):
    def __init__(
        self,
        name: str,
        buffer_class: str,
        word_bits: int,
        depth: Optional[int] = None,
        data_width: Optional[int] = None,
        meshX: Optional[int] = None,
        meshY: Optional[int] = None,
        instance: int = 1,
        block_size: Optional[int] = None,
        technology: int = 45,  # nm
        dram_type: Optional[str] = None,
        read_bandwidth: Optional[int] = None, # words per cycle
        write_bandwidth: Optional[int] = None, # words per cycle
        bandwidth: Optional[int] = None, # words per cycle
        cluster_size: Optional[int] = None,
        sizeKB: Optional[int] = None,
        width: Optional[int] = None,
    ):
        super().__init__()
        self.name = name
        self.buffer_class = buffer_class
        assert buffer_class in ["DRAM", "SRAM", "regfile"]
        self.width = width
        self.meshX = meshX
        if self.buffer_class in ["regfile"]:
            assert self.meshX is not None
        self.word_bits = word_bits
        self.depth = depth
        self.data_width = data_width
        self.meshY = meshY
        self.instance = instance
        self.block_size = block_size
        self.technology = technology
        self.dram_type = dram_type
        self.read_bandwidth = read_bandwidth
        self.write_bandwdith = write_bandwidth
        self.bandwidth = bandwidth
        self.cluster_size = cluster_size
        self.sizeKB = sizeKB
