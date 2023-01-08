from dominoc import ir
from typing import Any, List, Union, Tuple, Optional
from ..base import IRBase
from ..type_system.dtype import DTypeKind, DType
from .scalar_expr import *
from .stmt import *


__all__ = [
    "Block", "AttrBlock", "NdForBlock", "ForBlock",
    "BranchBlock", "SeqBlock", "SpatialBlock",
    "AtomBlock", "ReMapBlock", "NdAllocBlock", "AllocBlock",
    "_to_block"
]


class Block(ir.Block):
    def __init__(self):
        ir.Block.__init__(self)


def _to_block(block: Any):
    if isinstance(block, Block):
        return block
    if isinstance(block, Stmt):
        return AtomBlock(block)
    else:
        try:
            block = _to_stmt(block)
            return AtomBlock(block)
        except ValueError:
            raise ValueError(f"Can't convert {block} to Block.")


class AttrBlock(ir.AttrBlock, Block):
    def __init__(self, key: str, obj: IRBase, value: IRBase, body: Block):
        body = _to_block(body)
        ir.AttrBlock.__init__(self, key, obj, value, body)
        Block.__init__(self)


class NdForBlock(ir.NdForBlock, Block):
    def __init__(self, iters: List[Iterator], body: Block, compute_level: str):
        for it in iters:
            assert isinstance(it, Iterator)
        body = _to_block(body)
        if not isinstance(compute_level, str):
            assert isinstance(compute_level, ConstString)
            compute_level = compute_level.value
        ir.NdForBlock.__init__(self, iters, body, compute_level)
        Block.__init__(self)


class ForBlock(ir.ForBlock, Block):
    def __init__(self, iter: Iterator, body: Block, compute_level: str):
        assert isinstance(iter, Iterator)
        body = _to_block(body)
        if not isinstance(compute_level, str):
            assert isinstance(compute_level, ConstString)
            compute_level = compute_level.value
        ir.ForBlock.__init__(self, iter, body, compute_level)
        Block.__init__(self)


class BranchBlock(ir.BranchBlock, Block):
    def __init__(self, cond: Expr, true_branch: Block, false_branch: Block):
        cond = _to_expr(cond)
        true_branch = _to_block(true_branch)
        false_branch = _to_block(false_branch)
        ir.BranchBlock.__init__(self, cond, true_branch, false_branch)
        Block.__init__(self)


class SeqBlock(ir.SeqBlock, Block):
    def __init__(self, first: Block, second: Block):
        first = _to_block(first)
        second = _to_block(second)
        ir.SeqBlock.__init__(self, first, second)
        Block.__init__(self)


class SpatialBlock(ir.SpatialBlock, Block):
    def __init__(self, blocks: List[Block], spatial_bindings: List[ConstString]):
        blocks = [_to_block(x) for x in blocks]
        spatial_bindings = [ConstString(x) if isinstance(
            x, str) else x for x in spatial_bindings]
        for b in spatial_bindings:
            assert isinstance(b, ConstString)
        ir.SpatialBlock.__init__(self, blocks, spatial_bindings)
        Block.__init__(self)


class AtomBlock(ir.AtomBlock, Block):
    def __init__(self, stmt: Stmt):
        if not isinstance(stmt, Stmt):
            stmt = _to_stmt(stmt)
        ir.AtomBlock.__init__(self, stmt)
        Block.__init__(self)


class ReMapBlock(ir.ReMapBlock, Block):
    def __init__(self, mappings: List[MapVar], body: Block):
        for m in mappings:
            assert isinstance(m, MapVar)
        body = _to_block(body)
        ir.ReMapBlock.__init__(self, mappings, body)
        Block.__init__(self)


class NdAllocBlock(ir.NdAllocBlock, Block):
    def __init__(self, var: Var, shape: List[Expr], memory_scope: Union[ConstString, str], body: Block):
        assert isinstance(var, Var)
        shape = [_to_expr(x) for x in shape]
        if not isinstance(memory_scope, ConstString):
            assert isinstance(memory_scope, str)
            memory_scope = ConstString(memory_scope)
        body = _to_block(body)
        ir.NdAllocBlock.__init__(var, shape, memory_scope, body)
        Block.__init__(self)


class AllocBlock(ir.AllocBlock, Block):
    def __init__(self, var: Var, length: Expr, memory_scope: ConstString, body: Block):
        assert isinstance(var, Var)
        length = _to_expr(length)
        if not isinstance(memory_scope, ConstString):
            assert isinstance(memory_scope, str)
            memory_scope = ConstString(memory_scope)
        body = _to_block(body)
        ir.AllocBlock.__init__(var, length, memory_scope, body)
        Block.__init__(self)
