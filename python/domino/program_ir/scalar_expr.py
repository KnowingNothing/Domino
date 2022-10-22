import enum
from typing import Any, List, Union, Tuple
from ..base import IRBase
from ..type_system.dtype import DTypeKind, DType

# helper function
_dtype = DType.from_string


class ExprBase(IRBase):
    def __init__(self, dtype: DType):
        self.dtype: DType = dtype

    def is_const(self):
        return False

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


class BinExpr(ExprBase):
    def __init__(self, dtype: DType, a: ExprBase, b: ExprBase):
        super(BinExpr, self).__init__(dtype)
        assert a.dtype == b.dtype, (
            f"Binary Expr expects the same type for operands, but get {a.dtype} and {b.dtype}")
        self.a: ExprBase = a
        self.b: ExprBase = b


class UniExpr(ExprBase):
    def __init__(self, dtype: DType, a: ExprBase):
        super(UniExpr, self).__init__(dtype)
        self.a: ExprBase = a


class TerExpr(ExprBase):
    def __init__(self, dtype: DType, a: ExprBase, b: ExprBase, c: ExprBase):
        super(TerExpr, self).__init__(dtype)
        self.a: ExprBase = a
        self.b: ExprBase = b
        self.c: ExprBase = c


class ConstExpr(ExprBase):
    def __init__(self, dtype: DType, value: Any):
        super(ConstExpr, self).__init__(dtype)
        self.value: Any = value

    def is_const(self):
        return True

    def __str__(self):
        return str(self.value)


class MutableExpr(ExprBase):
    def __init__(self, dtype: DType):
        super(MutableExpr, self).__init__(dtype)


class MemRef(ExprBase):
    def __init__(self, var: "Var"):
        super().__init__(_dtype("mem_ref"))
        self.var: "Var" = var


##=-------------------------------------------------------------------=##
##
# Binary Operation IR Node
##
##=-------------------------------------------------------------------=##
class Add(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(Add, self).__init__(a.dtype, a, b)


class Sub(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(Sub, self).__init__(a.dtype, a, b)


class Mul(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(Mul, self).__init__(a.dtype, a, b)


class Div(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(Div, self).__init__(a.dtype, a, b)


class Mod(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(Mod, self).__init__(a.dtype, a, b)


class FloorDiv(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(FloorDiv, self).__init__(a.dtype, a, b)


class FloorMod(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(FloorMod, self).__init__(a.dtype, a, b)


class And(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(And, self).__init__(_dtype("bool"), a, b)


class Or(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(Or, self).__init__(_dtype("bool"), a, b)


class XOr(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(XOr, self).__init__(_dtype("bool"), a, b)


class BitAnd(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(BitAnd, self).__init__(a.dtype, a, b)


class BitOr(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(BitOr, self).__init__(a.dtype, a, b)


class BitXOr(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(BitXOr, self).__init__(a.dtype, a, b)


class GT(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(GT, self).__init__(_dtype("bool"), a, b)


class GE(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(GE, self).__init__(_dtype("bool"), a, b)


class LT(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(LT, self).__init__(_dtype("bool"), a, b)


class LE(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(LE, self).__init__(_dtype("bool"), a, b)


class EQ(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(EQ, self).__init__(_dtype("bool"), a, b)


class NE(BinExpr):
    def __init__(self, a: ExprBase, b: ExprBase):
        super(NE, self).__init__(_dtype("bool"), a, b)


##=-------------------------------------------------------------------=##
##
# Unary Operation IR Node
##
##=-------------------------------------------------------------------=##
class Cast(UniExpr):
    def __init__(self, dtype: DType, a: ExprBase):
        super(Cast, self).__init__(dtype, a)


class Broadcast(UniExpr):
    def __init__(self, a: ExprBase, lane: int):
        super().__init__(a.dtype.with_lanes(lane), a)


class Neg(UniExpr):
    def __init__(self, a: ExprBase):
        super(Neg, self).__init__(a.dtype, a)


class Not(UniExpr):
    def __init__(self, a: ExprBase):
        super(Not, self).__init__(_dtype("bool"), a)


class BitNot(UniExpr):
    def __init__(self, a: ExprBase):
        super(BitNot, self).__init__(a.dtype, a)


class Ceil(UniExpr):
    def __init__(self, dtype: DType, a: ExprBase):
        super(Ceil, self).__init__(dtype, a)
        assert dtype.is_int() or dtype.is_uint(
        ), f"Ceil output type should be Int or UInt, but get {dtype}"


class Floor(UniExpr):
    def __init__(self, dtype: DType, a: ExprBase):
        super(Floor, self).__init__(dtype, a)
        assert dtype.is_int() or dtype.is_uint(
        ), f"Floor output type should be Int or UInt, but get {dtype}"


##=-------------------------------------------------------------------=##
##
# Ternary Operation IR Node
##
##=-------------------------------------------------------------------=##
class Select(TerExpr):
    def __init__(self, cond: ExprBase, true_branch: ExprBase, false_branch: ExprBase):
        super(Select, self).__init__(
            true_branch.dtype, cond, true_branch, false_branch)
        assert (true_branch.dtype == false_branch.dtype), (
            f"Select Expr expects the same type for candidates, but get {true_branch.dtype} and {false_branch.dtype}")


##=-------------------------------------------------------------------=##
##
# Variable Operands Operation IR Node
##
##=-------------------------------------------------------------------=##
class CondAll(ExprBase):
    def __init__(self, phases: List[ExprBase]):
        super(CondAll, self).__init__(_dtype("bool"))
        self.phases = phases


class CondAny(ExprBase):
    def __init__(self, phases: List[ExprBase]):
        super(CondAny, self).__init__(_dtype("bool"))
        self.phases = phases


##=-------------------------------------------------------------------=##
##
# Constant IR Node
##
##=-------------------------------------------------------------------=##
class ConstInt(ConstExpr):
    def __init__(self, value: int, bits: int = 32, lane: int = 1):
        super(ConstInt, self).__init__(DType(DTypeKind.Int, bits, lane), value)


class ConstUInt(ConstExpr):
    def __init__(self, value: int, bits: int = 32, lane: int = 1):
        super(ConstUInt, self).__init__(
            DType(DTypeKind.UInt, bits, lane), value)


class ConstFloat(ConstExpr):
    def __init__(self, value: float, bits: int = 32, lane: int = 1):
        super(ConstFloat, self).__init__(
            DType(DTypeKind.Float, bits, lane), value)


class ConstBFloat(ConstExpr):
    def __init__(self, value: float, bits: int = 16, lane: int = 1):
        super(ConstBFloat, self).__init__(
            DType(DTypeKind.BFloat, bits, lane), value)


class ConstTFloat(ConstExpr):
    def __init__(self, value: float, bits: int = 32, lane: int = 1):
        super(ConstTFloat, self).__init__(
            DType(DTypeKind.TFloat, bits, lane), value)


class ConstString(ConstExpr):
    def __init__(self, value: str):
        super(ConstString, self).__init__(_dtype("string"), value)


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
# Range IR Node
##
##=-------------------------------------------------------------------=##
class Range(ExprBase):
    def __init__(self, beg: ExprBase, extent: ExprBase, step: ExprBase):
        super(Range, self).__init__(_dtype("ignore"))
        self.beg = beg
        self.extent = extent
        self.step = step

    def __str__(self) -> str:
        return f"Range(beg={self.beg},ext={self.extent},step={self.step})"


class ExprList(ExprBase):
    def __init__(self, args: List[ExprBase]):
        super(ExprList, self).__init__(_dtype("ignore"))
        self.value_list = args

    def __str__(self):
        return "[" + ",".join([str(x) for x in self.value_list]) + "]"


##=-------------------------------------------------------------------=##
##
# Mutable IR Node
##
##=-------------------------------------------------------------------=##
class Var(MutableExpr):
    def __init__(self, dtype: Union[DType, str], name: str = ""):
        super(Var, self).__init__(_dtype(dtype)
                                  if isinstance(dtype, str) else dtype)
        self.name = name


##=-------------------------------------------------------------------=##
##
# Iterator IR Node
##
##=-------------------------------------------------------------------=##
class IterTypeKind(enum.Enum):
    Spatial = 0
    Reduce = 1


class Iterator(ExprBase):
    def __init__(self, var: Var, range: Union[Range, List, Tuple], iter_type: IterTypeKind):
        super(Iterator, self).__init__(var.dtype)
        self.var = var
        if isinstance(range, Range):
            self.range = range
        else:
            if len(range) == 2:
                self.range = Range(
                    ConstInt(range[0]), ConstInt(range[1]), ConstInt(1))
            elif len(range) == 3:
                self.range = Range(ConstInt(range[0]), ConstInt(
                    range[1]), ConstInt(range[2]))
            else:
                raise RuntimeError(
                    "Range should be [beg, extent] or [beg, extent, step].")
        self.iter_type = iter_type


##=-------------------------------------------------------------------=##
##
# Load IR Node
##
##=-------------------------------------------------------------------=##
class NdLoad(ExprBase):
    def __init__(self, mem_ref: MemRef, indices: List[ExprBase]):
        super(NdLoad, self).__init__(mem_ref.var.dtype)
        self.mem_ref = mem_ref
        self.indices = indices


class Load(ExprBase):
    def __init__(self, mem_ref: MemRef, addr: ExprBase):
        super(Load, self).__init__(mem_ref.var.dtype)
        self.mem_ref = mem_ref
        self.addr = addr


##=-------------------------------------------------------------------=##
##
# Map IR Node
##
##=-------------------------------------------------------------------=##
class MapVar(ExprBase):
    def __init__(self, name: str, expr: ExprBase):
        super(MapVar, self).__init__(expr.dtype)
        self.var = Var(expr.dtype, name)
        self.expr = expr


##=-------------------------------------------------------------------=##
##
# Memory Reference IR Node
##
##=-------------------------------------------------------------------=##
class MemSlice(MemRef):
    def __init__(self, var: Var, slices: List[Range]):
        super(MemSlice, self).__init__(var)
        self.slices: List[Range] = slices


##=-------------------------------------------------------------------=##
##
# Call IR Node
##
##=-------------------------------------------------------------------=##
class Call(ExprBase):
    def __init__(self, dtype: DType, name: str, args: List[Union[ExprBase, str]]):
        super(Call, self).__init__(dtype)
        self.name: str = name
        self.args: List[Union[ExprBase, str]] = args
