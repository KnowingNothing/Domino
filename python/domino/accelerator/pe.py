from ..base import HardwareLevel
from typing import Optional


class ALU(HardwareLevel):
    def __init__(self, name: str, alu_class: str, meshX: int, datawidth: int, meshY: Optional[int] = None, instance: int = 1):
        super().__init__()
        self.name = name
        self.alu_class = alu_class
        self.meshX = meshX
        self.datawidth = datawidth
        self.meshY = meshY
        self.instance = instance


class Engine(HardwareLevel):
    def __init__(self, name: str, technology: Optional[str] = None, instance: int = 1, meshX: Optional[int] = None, meshY: Optional[int] = None):
        super().__init__()
        self.name = name
        self.technology = technology
        self.instance = instance
        self.local = []
        self.meshX = meshX
        self.meshY = meshY

    def add_local(self, *level):
        if len(level) == 0:
            return
        if len(level) == 1:
            if isinstance(level[0], list):
                self.local.extend(level[0])
            else:
                self.local.append(level[0])
        else:
            for l in level:
                assert isinstance(l, HardwareLevel)
                self.local.append(l)
