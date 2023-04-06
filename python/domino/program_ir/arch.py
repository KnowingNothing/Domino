from dominoc import ir
from typing import Any, List
from .block import Block, _to_block
from .scalar_expr import Var, _to_expr


__all__ = ["Arch", "MemoryLevel", "ComputeLevel"]


class Arch(ir.Arch):
    def __init__(self):
        ir.Arch.__init__(self)

    def is_memory_level(self):
        return isinstance(self, ir.MemoryLevel)

    def is_compute_level(self):
        return isinstance(self, ir.ComputeLevel)


class MemoryLevel(ir.MemoryLevel, Arch):
    def __init__(self, level: int, block: Block, sub_levels: List[Arch]):
        level = _to_expr(level)
        block = _to_block(block)
        ir.MemoryLevel.__init__(self, level, block, sub_levels)
        Arch.__init__(self)

    def set_scope(self, scope: str):
        self.scope = scope

    def set_annotation(self, annotation: str):
        self.annotation = annotation
        
    def set_split_point(self, split_point: int):
        self.split_point = split_point


class ComputeLevel(ir.ComputeLevel, Arch):
    def __init__(self, level: int, block: Block, sub_levels: List[Arch]):
        level = _to_expr(level)
        block = _to_block(block)
        ir.ComputeLevel.__init__(self, level, block, sub_levels)
        Arch.__init__(self)

    def set_produce_var(self, var: Var):
        self.produce_var = var
