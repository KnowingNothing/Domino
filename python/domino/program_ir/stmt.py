from dominoc import ir
from typing import Any, List, Union, Tuple
from ..base import IRBase
from ..type_system.dtype import DTypeKind, DType
from .scalar_expr import *


__all__ = [
    "Stmt", "NdStore", "Store", "Evaluate", "_to_stmt"
]


class Stmt(ir.Stmt):
    def __init__(self):
        ir.Stmt.__init__(self)


def _to_stmt(stmt: Any):
    if isinstance(stmt, ir.Stmt):
        return stmt
    if isinstance(stmt, ir.Expr):
        return Evaluate(stmt)
    else:
        try:
            stmt = _to_expr(stmt)
            return Evaluate(stmt)
        except ValueError:
            raise ValueError(f"Can't convert {stmt} to Stmt")


class NdStore(ir.NdStore, Stmt):
    def __init__(self, mem_ref: MemRef, indices: Union[ExprList, List[Any]], value: Expr):
        indices = _to_expr(indices)
        value = _to_expr(value)
        ir.NdStore.__init__(self, mem_ref, indices, value)
        Stmt.__init__(self)


class Store(ir.Store, Stmt):
    def __init__(self, mem_ref: MemRef, addr: Expr, value: Expr):
        addr = _to_expr(addr)
        value = _to_expr(value)
        ir.Store.__init__(self, mem_ref, addr, value)
        Stmt.__init__(self)


class Evaluate(ir.Evaluate, Stmt):
    def __init__(self, expr: Expr):
        expr = _to_expr(expr)
        ir.Evaluate.__init__(self, expr)
        Stmt.__init__(self)
