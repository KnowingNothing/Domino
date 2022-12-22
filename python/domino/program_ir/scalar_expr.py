import enum
from dominoc import ir
from typing import Any, List, Union, Tuple
from ..base import IRBase
from ..type_system.dtype import DTypeKind, DType

# helper function
_dtype = DType.make

__all__ = [
    "Expr", "BinExpr", "UniExpr", "TerExpr", "ConstExpr", "MutableExpr", "MemRef",
    "Add", "Sub", "Mul", "Div", "Mod", "FloorDiv", "FloorMod", "And", "Or",
    "XOr", "BitAnd", "BitOr", "BitXOr", "GT", "GE", "LT", "LE", "EQ", "NE",
    "Cast", "Broadcast", "Neg", "Not", "BitNot", "Ceil", "Floor",
    "Select",
    "Range", "ExprList",
    "CondAll", "CondAny",
    "ConstInt", "ConstUInt", "ConstFloat", "ConstBFloat", "ConstTFloat", "ConstString", "make_const",
    "Var", "Iterator", "NdLoad", "Load", "MapVar", "Slice", "MemSlice", "Call"
]


class Expr(ir.Expr):
    def __init__(self, dtype: DType):
        super(Expr, self).__init__(dtype)

    def __mul__(self, others):
        if isinstance(others, self.__class__):
            return Mul(self, others)
        elif isinstance(others, int):
            return Mul(self, ConstInt(others))
        elif isinstance(others, float):
            return Mul(self, ConstFloat(others))
        else:
            raise RuntimeError(f"Can't perform {self} * {others}")

    def __rmul__(self, others):
        if isinstance(others, self.__class__):
            return Mul(others, self)
        elif isinstance(others, int):
            return Mul(ConstInt(others), self)
        elif isinstance(others, float):
            return Mul(ConstFloat(others), self)
        else:
            raise RuntimeError(f"Can't perform {others} * {self}")


##=-------------------------------------------------------------------=##
##
# Non-Terminal IR Node
##
##=-------------------------------------------------------------------=##


class BinExpr(ir.BinExpr, Expr):
    def __init__(self, dtype: DType, a: Expr, b: Expr):
        ir.BinExpr.__init__(self, dtype, a, b)
        Expr.__init__(self, dtype)


class UniExpr(ir.UniExpr, Expr):
    def __init__(self, dtype: DType, a: Expr):
        ir.UniExpr.__init__(self, dtype, a)
        Expr.__init__(self, dtype)


class TerExpr(ir.TerExpr, Expr):
    def __init__(self, dtype: DType, a: Expr, b: Expr, c: Expr):
        ir.TerExpr.__init__(dtype, a, b, c)
        Expr.__init__(self, dtype)


class ConstExpr(ir.ConstExpr, Expr):
    def __init__(self, dtype: DType):
        ir.ConstExpr.__init__(self, dtype)
        Expr.__init__(self, dtype)


class MutableExpr(ir.MutableExpr, Expr):
    def __init__(self, dtype: DType):
        ir.MutableExpr.__init__(self, dtype)
        Expr.__init__(self, dtype)


class MemRef(ir.MemRef, Expr):
    def __init__(self, var: "Var", offset: Expr):
        ir.MemRef.__init__(self, var, offset)
        Expr.__init__(self, _dtype("mem_ref"))

##=-------------------------------------------------------------------=##
##
# Binary Operation IR Node
##
##=-------------------------------------------------------------------=##


class Add(ir.Add, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.Add.__init__(self, a, b)
        BinExpr.__init__(self, a.dtype, a, b)


class Sub(ir.Sub, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.Sub.__init__(self, a, b)
        BinExpr.__init__(self, a.dtype, a, b)


class Mul(ir.Mul, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.Mul.__init__(self, a, b)
        BinExpr.__init__(self, a.dtype, a, b)


class Div(ir.Div, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.Div.__init__(self, a, b)
        BinExpr.__init__(self, a.dtype, a, b)


class Mod(ir.Mod, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.Mod.__init__(self, a, b)
        BinExpr.__init__(self, a.dtype, a, b)


class FloorDiv(ir.FloorDiv, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.FloorDiv.__init__(self, a, b)
        BinExpr.__init__(self, a.dtype, a, b)


class FloorMod(ir.FloorMod, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.FloorMod.__init__(self, a, b)
        BinExpr.__init__(self, a.dtype, a, b)


class And(ir.And, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.And.__init__(self, a, b)
        BinExpr.__init__(self, a.dtype, a, b)


class Or(ir.Or, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.Or.__init__(self, a, b)
        BinExpr.__init__(self, self.dtype, a, b)


class XOr(ir.XOr, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.XOr.__init__(self, a, b)
        BinExpr.__init__(self, self.dtype, a, b)


class BitAnd(ir.BitAnd, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.BitAnd.__init__(self, a, b)
        BinExpr.__init__(self, self.dtype, a, b)


class BitOr(ir.BitOr, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.BitOr.__init__(self, a, b)
        BinExpr.__init__(self, self.dtype, a, b)


class BitXOr(ir.BitXOr, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.BitXOr.__init__(self, a, b)
        BinExpr.__init__(self, self.dtype, a, b)


class GT(ir.GT, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.GT.__init__(self, a, b)
        BinExpr.__init__(self, self.dtype, a, b)


class GE(ir.GE, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.GE.__init__(self, a, b)
        BinExpr.__init__(self, self.dtype, a, b)


class LT(ir.LT, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.LT.__init__(self, a, b)
        BinExpr.__init__(self, self.dtype, a, b)


class LE(ir.LE, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.LE.__init__(self, a, b)
        BinExpr.__init__(self, self.dtype, a, b)


class EQ(ir.EQ, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.EQ.__init__(self, a, b)
        BinExpr.__init__(self, self.dtype, a, b)


class NE(ir.NE, BinExpr):
    def __init__(self, a: Expr, b: Expr):
        ir.NE.__init__(self, a, b)
        BinExpr.__init__(self, self.dtype, a, b)


##=-------------------------------------------------------------------=##
##
# Unary Operation IR Node
##
##=-------------------------------------------------------------------=##
class Cast(ir.Cast, UniExpr):
    def __init__(self, dtype: DType, a: Expr):
        ir.Cast.__init__(self, a)
        UniExpr.__init__(self, self.dtype, a)


class Broadcast(ir.Broadcast, UniExpr):
    def __init__(self, a: Expr, lane: int):
        ir.Broadcast.__init__(self, a)
        UniExpr.__init__(self, self.dtype, a)


class Neg(ir.Neg, UniExpr):
    def __init__(self, a: Expr):
        ir.Neg.__init__(self, a)
        UniExpr.__init__(self, self.dtype, a)


class Not(ir.Not, UniExpr):
    def __init__(self, a: Expr):
        ir.Not.__init__(self, a)
        UniExpr.__init__(self, self.dtype, a)


class BitNot(ir.BitNot, UniExpr):
    def __init__(self, a: Expr):
        ir.BitNot.__init__(self, a)
        UniExpr.__init__(self, self.dtype, a)


class Ceil(ir.Ceil, UniExpr):
    def __init__(self, dtype: DType, a: Expr):
        ir.Ceil.__init__(self, a)
        UniExpr.__init__(self, self.dtype, a)


class Floor(ir.Floor, UniExpr):
    def __init__(self, dtype: DType, a: Expr):
        ir.Floor.__init__(self, a)
        UniExpr.__init__(self, self.dtype, a)


##=-------------------------------------------------------------------=##
##
# Ternary Operation IR Node
##
##=-------------------------------------------------------------------=##
class Select(ir.Select, TerExpr):
    def __init__(self, cond: Expr, true_branch: Expr, false_branch: Expr):
        ir.Select.__init__(self, cond, true_branch, false_branch)
        TerExpr.__init__(self, self.dtype, cond, true_branch, false_branch)


##=-------------------------------------------------------------------=##
##
# Range IR Node
##
##=-------------------------------------------------------------------=##
class Range(ir.Range, Expr):
    def __init__(self, beg: Expr, extent: Expr, step: Expr):
        ir.Range.__init__(self, beg, extent, step)
        Expr.__init__(self, self.dtype)


class ExprList(ir.ExprList, Expr):
    def __init__(self, args: List[Expr]):
        ir.ExprList.__init__(self, args)
        Expr.__init__(self, self.dtype)

    def __getitem__(self, idx):
        assert idx < len(self.value_list), "Index out of range."
        return self.value_list[idx]


##=-------------------------------------------------------------------=##
##
# Variable Operands Operation IR Node
##
##=-------------------------------------------------------------------=##
class CondAll(ir.CondAll, Expr):
    def __init__(self, phases: List[Expr]):
        ir.CondAll.__init__(self, phases)
        Expr.__init__(self, self.dtype)


class CondAny(ir.CondAny, Expr):
    def __init__(self, phases: List[Expr]):
        ir.CondAny.__init__(self, phases)
        Expr.__init__(self, self.dtype)


##=-------------------------------------------------------------------=##
##
# Constant IR Node
##
##=-------------------------------------------------------------------=##
class ConstInt(ir.ConstInt, ConstExpr):
    def __init__(self, value: int, bits: int = 32, lane: int = 1):
        ir.ConstInt.__init__(self, value, bits, lane)
        ConstExpr.__init__(self, self.dtype)


class ConstUInt(ir.ConstUInt, ConstExpr):
    def __init__(self, value: int, bits: int = 32, lane: int = 1):
        ir.ConstUInt.__init__(self, value, bits, lane)
        ConstExpr.__init__(self, self.dtype)


class ConstFloat(ir.ConstFloat, ConstExpr):
    def __init__(self, value: float, bits: int = 32, lane: int = 1):
        ir.ConstFloat.__init__(self, value, bits, lane)
        ConstExpr.__init__(self, self.dtype)


class ConstBFloat(ir.ConstBFloat, ConstExpr):
    def __init__(self, value: float, bits: int = 16, lane: int = 1):
        ir.ConstBFloat.__init__(self, value, bits, lane)
        ConstExpr.__init__(self, self.dtype)


class ConstTFloat(ir.ConstTFloat, ConstExpr):
    def __init__(self, value: float, bits: int = 32, lane: int = 1):
        ir.ConstTFloat.__init__(self, value, bits, lane)
        ConstExpr.__init__(self, self.dtype)


class ConstString(ir.ConstString, ConstExpr):
    def __init__(self, value: str):
        ir.ConstString.__init__(self, value)
        ConstExpr.__init__(self, self.dtype)


def make_const(value, dtype):
    dtype = DType.make(dtype)
    if dtype.is_int():
        return ConstInt(value, dtype.bits, dtype.lane)
    elif dtype.is_uint():
        return ConstUInt(value, dtype.bits, dtype.lane)
    elif dtype.is_float():
        return ConstFloat(value, dtype.bits, dtype.lane)
    elif dtype.is_bfloat():
        return ConstBFloat(value, dtype.bits, dtype.lane)
    elif dtype.is_tfloat():
        return ConstTFloat(value, dtype.bits, dtype.lane)
    elif dtype.is_string():
        return ConstString(value)
    else:
        raise NotImplementedError(f"Can't make const {dtype}.")


##=-------------------------------------------------------------------=##
##
# Mutable IR Node
##
##=-------------------------------------------------------------------=##
class Var(ir.Var, MutableExpr):
    def __init__(self, dtype: Union[DType, str], name: str = ""):
        ir.Var.__init__(self, dtype, name)
        MutableExpr.__init__(self, self.dtype)


##=-------------------------------------------------------------------=##
##
# Iterator IR Node
##
##=-------------------------------------------------------------------=##
# class IterTypeKind(enum.Enum):
#     Spatial = 0
#     Reduce = 1

IterTypeKind = ir.IterTypeKind


class Iterator(ir.Iterator, Expr):
    def __init__(self, var: Var, range: Union[Range, List, Tuple], iter_type: IterTypeKind):
        if not isinstance(range, Range):
            if len(range) == 2:
                range = Range(
                    ConstInt(range[0]), ConstInt(range[1]), ConstInt(1))
            elif len(range) == 3:
                range = Range(ConstInt(range[0]), ConstInt(
                    range[1]), ConstInt(range[2]))
            else:
                raise RuntimeError(
                    "Range should be [beg, extent] or [beg, extent, step].")
        ir.Iterator.__init__(self, var, range, iter_type)
        Expr.__init__(self, self.dtype)


##=-------------------------------------------------------------------=##
##
# Load IR Node
##
##=-------------------------------------------------------------------=##
class NdLoad(ir.NdLoad, Expr):
    def __init__(self, mem_ref: MemRef, indices: List[Expr]):
        ir.NdLoad.__init__(self, mem_ref, indices)
        Expr.__init__(self, self.dtype)


class Load(ir.Load, Expr):
    def __init__(self, mem_ref: MemRef, addr: Expr):
        ir.Load.__init__(self, mem_ref, addr)
        Expr.__init__(self, self.dtype)


##=-------------------------------------------------------------------=##
##
# Map IR Node
##
##=-------------------------------------------------------------------=##
class MapVar(ir.MapVar, Expr):
    def __init__(self, name: str, expr: Expr):
        ir.MapVar.__init__(self, name, expr)
        Expr.__init__(self, self.dtype)


##=-------------------------------------------------------------------=##
##
# Memory Reference IR Node
##
##=-------------------------------------------------------------------=##
class Slice(ir.Slice, Expr):
    def __init__(self, indices: List[Range]):
        ir.Slice.__init__(self, indices)
        Expr.__init__(self, self.dtype)


class MemSlice(ir.MemSlice, MemRef):
    def __init__(self, var: Var, offset: Expr, slices: Union[Slice, List[Range]]):
        if isinstance(slices, Slice):
            ir.MemSlice.__init__(self, var, offset, slices)
        else:
            ir.MemSlice.__init__(self, var, offset, Slice(slices))
        MemRef.__init__(self, var, offset)


##=-------------------------------------------------------------------=##
##
# Call IR Node
##
##=-------------------------------------------------------------------=##
class Call(ir.Call, Expr):
    def __init__(self, dtype: DType, name: str, args: List[Union[Expr, str]]):
        new_args = [arg if isinstance(
            arg, Expr) else ConstString(arg) for arg in args]
        ir.Call.__init__(self, dtype, name, ExprList(new_args))
        Expr.__init__(self, self.dtype)
