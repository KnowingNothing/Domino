from dominoc import ir
from ..type_system import DType
from .scalar_expr import *
from .functional import print_ir
from .block import _to_block
from ..passes import (get_input_tensor_vars, ProdConsumGraph,
                      get_input_tensor_indices)
from typing import List, Union, Any, Optional


__all__ = ["NameGenerator", "ReduceOp", "ElemOp", "Tensor", "TensorView", "ConstTensor", "Loop", "SpatialLoop",
           "ReduceLoop", "TensorizedLoop", "SLoop", "RLoop", "TLoop", "make_prod_consum_graph",
           "max", "cast", "pack_value", "clip", "sqrt", "exp", "make_const", "const", ]


class NameGenerator(object):
    name_cache = {}

    @classmethod
    def gen_name(cls, hint: str):
        parts = hint.split("_")
        if parts[0] not in cls.name_cache:
            cls.name_cache[parts[0]] = 0
            return parts[0] + f"_{0}"
        else:
            v = cls.name_cache[parts[0]] + 1
            cls.name_cache[parts[0]] += 1
            return parts[0] + f"_{v}"


class ComputeOp(object):
    pass


class ReduceOp(ComputeOp):
    def __init__(self, kind: str) -> None:
        assert kind in ["sum", "max", "min"]
        self.kind = kind

    def __str__(self):
        return self.kind


class ElemOp(ComputeOp):
    def __str__(self):
        return "elem"


class Tensor(object):
    tensor_cache = {}

    def __init__(
            self,
            shape: List[Union[int, Expr]],
            name: str = "T",
            dtype: Union[DType, str] = "float32"):
        self.shape = shape
        self.name = NameGenerator.gen_name(name)
        self.dtype = dtype
        self.var = Var(self.dtype, self.name)
        self.init: Optional[Compute] = None
        self.updates: List[Compute] = []
        # FIXME: how to avoid global tensor cache?
        self.tensor_cache[self.var] = self

    def is_const(self):
        return False

    def _parse_keys(self, keys, simple_indices=False):
        new_keys = []
        new_shape = []
        for k, s in zip(keys, self.shape):
            if isinstance(k, Loop):
                new_keys.append(k.var)
            elif isinstance(k, slice):
                if k.start is None:
                    start = _to_expr(0)
                elif isinstance(k.start, Loop):
                    start = k.start.var
                else:
                    start = _to_expr(k.start)

                if k.step is None:
                    step = _to_expr(1)
                elif isinstance(k.step, Loop):
                    step = k.step.var
                else:
                    step = _to_expr(k.step)

                if k.stop is None:
                    stop = _to_expr(s)
                elif isinstance(k.stop, Loop):
                    stop = k.stop.var
                else:
                    stop = _to_expr(k.stop)

                new_keys.append(slice(start, stop, step))
                new_shape.append((stop - start)//step)
            else:
                if simple_indices:
                    raise RuntimeError(
                        f"Expect simple indices but get {print_ir(_to_expr(k), print_out=False)}")
                new_keys.append(_to_expr(k))
        return new_shape, new_keys

    def __getitem__(self, keys):
        new_shape, new_keys = self._parse_keys(keys)
        return TensorView(new_shape, self, new_keys)

    def __setitem__(self, keys, value):
        new_shape, new_keys = self._parse_keys(keys, simple_indices=True)
        view = TensorView(new_shape, self, new_keys)
        if isinstance(value, TensorView):
            value = value.as_expr()
        else:
            value = _to_expr(value)
        compute = Compute(view, value, ElemOp(), [])
        if self.init is None:
            self.init = compute
        else:
            self.updates.append(compute)

    def Init(self, keys, value):
        assert self.init is None
        new_shape, new_keys = self._parse_keys(keys, simple_indices=True)
        view = TensorView(new_shape, self, new_keys)
        if isinstance(value, TensorView):
            value = value.as_expr()
        else:
            value = _to_expr(value)
        compute = Compute(view, value, ElemOp(), [])
        self.init = compute

    def Update(self, keys, value, op: "ComputeOp" = ElemOp(), reduce_axis: Optional[List["Loop"]] = None):
        new_shape, new_keys = self._parse_keys(keys, simple_indices=True)
        reduce_axis = [] if reduce_axis is None else [x.var for x in reduce_axis]
        if isinstance(op, ReduceOp):
            assert len(
                reduce_axis) > 0, "ReduceOp expects at least 1 reduce_axis."
        view = TensorView(new_shape, self, new_keys)
        if isinstance(value, TensorView):
            value = value.as_expr()
        else:
            value = _to_expr(value)
        compute = Compute(view, value, op, reduce_axis)
        self.updates.append(compute)

    def __str__(self):
        shape_str = ",".join([print_ir(x, print_out=False) if isinstance(
            x, Expr) else str(x) for x in self.shape])
        return f"Tensor({self.name}, [{shape_str}], {self.dtype})"

    def __repr__(self):
        return str(self)

    @classmethod
    def _gen_name(cls, hint: str):
        parts = hint.split("_")
        if parts[0] not in cls.name_cache:
            cls.name_cache[parts[0]] = 0
            return parts[0] + f"_{0}"
        else:
            v = cls.name_cache[parts[0]] + 1
            cls.name_cache[parts[0]] += 1
            return parts[0] + f"_{v}"


class ConstTensor(Tensor):
    def __init__(
            self,
            shape: List[Union[int, Expr]],
            name: str = "",
            dtype: Union[DType, str] = "float32"):
        super(ConstTensor, self).__init__(shape, name, dtype)
        self.var = ConstVar(self.dtype, self.name)

    def is_const(self):
        return True


class TensorView(object):
    def __init__(self, shape: List[Expr], tensor_or_view: Union[Tensor, "TensorView"], indices: List[Expr]) -> None:
        self.shape = shape
        self.tensor_or_view = tensor_or_view
        self.indices = indices

        if not isinstance(tensor_or_view, Tensor):
            simplified_view = self._resolve(self.shape, self.indices)
            self.shape = simplified_view.shape
            self.tensor_or_view = simplified_view.tensor_or_view
            self.indices = simplified_view.indices
        self.name = f"view_{self.tensor_or_view.name}"

    def _parse_keys(self, keys):
        new_keys = []
        new_shape = []
        for k, s in zip(keys, self.shape):
            if isinstance(k, Loop):
                new_keys.append(k.var)
            elif isinstance(k, slice):
                if k.start is None:
                    start = _to_expr(0)
                elif isinstance(k.start, Loop):
                    start = k.start.var
                else:
                    start = _to_expr(k.start)

                if k.step is None:
                    step = _to_expr(1)
                elif isinstance(k.step, Loop):
                    step = k.step.var
                else:
                    step = _to_expr(k.step)

                if k.stop is None:
                    stop = _to_expr(s)
                elif isinstance(k.stop, Loop):
                    stop = k.stop.var
                else:
                    stop = _to_expr(k.stop)

                new_keys.append(slice(start, stop, step))
                new_shape.append((stop - start)//step)
            else:
                new_keys.append(_to_expr(k))
        return new_shape, new_keys

    def __getitem__(self, keys):
        new_shape, new_keys = self._parse_keys(keys)
        return TensorView(new_shape, self, new_keys)

    def _resolve(self, shape, indices):
        if isinstance(self.tensor_or_view, Tensor):
            return TensorView(shape, self.tensor_or_view, indices)
        else:
            idx = 0
            new_indices = []
            for ind in self.tensor_or_view.indices:
                if isinstance(ind, slice):
                    if isinstance(self.indices[idx], slice):
                        start = ind.start + self.indices[idx].start * ind.step
                        step = ind.step * self.indices[idx].step
                        stop = ind.start + self.indices[idx].stop * ind.step
                        new_indices.append(slice(start, stop, step))
                    else:
                        start = ind.start + self.indices[idx] * ind.step
                        new_indices.append(start)
                    idx += 1
                else:
                    new_indices.append(ind)
            return self.tensor_or_view._resolve(shape, new_indices)

    def is_scalar(self):
        return len(self.shape) == 0

    def as_expr(self):
        if self.is_scalar():
            return NdLoad(MemRef(self.tensor_or_view.var, 0), self.indices)
        else:
            raise RuntimeError("Can't convert TensorView with slice to Expr.")

    def __add__(self, other):
        if isinstance(other, TensorView):
            if self.is_scalar() and other.is_scalar():
                left = NdLoad(MemRef(self.tensor_or_view.var, 0), self.indices)
                right = NdLoad(
                    MemRef(other.tensor_or_view.var, 0), other.indices)
                return left + right
            else:
                raise RuntimeError(
                    f"Only support scalar arithmetic for now, but the TensorViews are {self} and {other}")
        else:
            try:
                other = _to_expr(other)
                if self.is_scalar():
                    ndload = NdLoad(
                        MemRef(self.tensor_or_view.var, 0), self.indices)
                    return ndload + other
                else:
                    raise RuntimeError(
                        f"Only support scalar arithmetic for now, but the TensorView is {self}")
            except Exception as e:
                raise ValueError(
                    f"Can't perform TensorView * {type(other)} (i.e., {other})")

    def __sub__(self, other):
        if isinstance(other, TensorView):
            if self.is_scalar() and other.is_scalar():
                left = NdLoad(MemRef(self.tensor_or_view.var, 0), self.indices)
                right = NdLoad(
                    MemRef(other.tensor_or_view.var, 0), other.indices)
                return left - right
            else:
                raise RuntimeError(
                    f"Only support scalar arithmetic for now, but the TensorViews are {self} and {other}")
        else:
            try:
                other = _to_expr(other)
                if self.is_scalar():
                    ndload = NdLoad(
                        MemRef(self.tensor_or_view.var, 0), self.indices)
                    return ndload - other
                else:
                    raise RuntimeError(
                        f"Only support scalar arithmetic for now, but the TensorView is {self}")
            except Exception as e:
                raise ValueError(
                    f"Can't perform TensorView * {type(other)} (i.e., {other})")

    def __mul__(self, other):
        if isinstance(other, TensorView):
            if self.is_scalar() and other.is_scalar():
                left = NdLoad(MemRef(self.tensor_or_view.var, 0), self.indices)
                right = NdLoad(
                    MemRef(other.tensor_or_view.var, 0), other.indices)
                return left * right
            else:
                raise RuntimeError(
                    f"Only support scalar arithmetic for now, but the TensorViews are {self} and {other}")
        else:
            try:
                other = _to_expr(other)
                if self.is_scalar():
                    ndload = NdLoad(
                        MemRef(self.tensor_or_view.var, 0), self.indices)
                    return ndload * other
                else:
                    raise RuntimeError(
                        f"Only support scalar arithmetic for now, but the TensorView is {self}")
            except Exception as e:
                raise ValueError(
                    f"Can't perform TensorView * {type(other)} (i.e., {other})")

    def __truediv__(self, other):
        if isinstance(other, TensorView):
            if self.is_scalar() and other.is_scalar():
                left = NdLoad(MemRef(self.tensor_or_view.var, 0), self.indices)
                right = NdLoad(
                    MemRef(other.tensor_or_view.var, 0), other.indices)
                return left / right
            else:
                raise RuntimeError(
                    f"Only support scalar arithmetic for now, but the TensorViews are {self} and {other}")
        else:
            try:
                other = _to_expr(other)
                if self.is_scalar():
                    ndload = NdLoad(
                        MemRef(self.tensor_or_view.var, 0), self.indices)
                    return ndload / other
                else:
                    raise RuntimeError(
                        f"Only support scalar arithmetic for now, but the TensorView is {self}")
            except Exception as e:
                raise ValueError(
                    f"Can't perform TensorView * {type(other)} (i.e., {other})")

    def __str__(self):
        shape = ",".join([print_ir(x, print_out=False) for x in self.shape])
        indices = []
        for x in self.indices:
            if isinstance(x, slice):
                indices.append(
                    f"{print_ir(x.start, print_out=False)}:{print_ir(x.stop, print_out=False)}:{print_ir(x.step, print_out=False)}")
            else:
                indices.append(print_ir(x, print_out=False))
        indices = ",".join(indices)
        return f"TensorView(tensor={self.tensor_or_view.name}, shape=[{shape}], [{indices}])"


class Compute(object):
    def __init__(self, tensor_view: TensorView, value: Expr, operation: ComputeOp, reduce_axis: List[Var]) -> None:
        self.tensor_view = tensor_view
        assert isinstance(value, Expr)
        self.value = value
        self.operation = operation
        self.reduce_axis = reduce_axis

    def input_tensors(self):
        vars = get_input_tensor_vars(self.value)
        visit = set()
        ret = []
        for v in vars:
            if v not in visit:
                ret.append(Tensor.tensor_cache[v])
                visit.add(v)
        return ret

    def get_tensor_indices(self, tensor: Union[Tensor, ir.Var]):
        if isinstance(tensor, ir.Var):
            tensor = Tensor.tensor_cache[tensor]
        assert isinstance(tensor, Tensor)
        return get_input_tensor_indices(self.value, tensor.var)

    def all_loops(self):
        visit = set()
        ret = []
        for l in self.tensor_view.indices:
            if l not in visit:
                assert l in Loop.loop_cache, f"The index {print_ir(l, print_out=False)} is not a simple Var or the loop Var is not cached for some reason."
                ret.append(Loop.loop_cache[l])
                visit.add(l)
        for l in self.reduce_axis:
            if l not in visit:
                assert l in Loop.loop_cache, f"The index {print_ir(l, print_out=False)} is not a simple Var or the loop Var is not cached for some reason."
                ret.append(Loop.loop_cache[l])
                visit.add(l)
        return ret

    def reduce_loops(self):
        return [Loop.loop_cache[v] for v in self.reduce_axis]

    def spatial_loops(self):
        visit = set()
        ret = []
        for l in self.tensor_view.indices:
            if l not in visit:
                assert l in Loop.loop_cache, f"The index {print_ir(l, print_out=False)} is not a simple Var or the loop Var is not cached for some reason."
                ret.append(Loop.loop_cache[l])
                visit.add(l)
        return ret

    def has_reduce(self):
        return isinstance(self.operation, ReduceOp)

    def __str__(self):
        if isinstance(self.operation, ReduceOp):
            if self.operation.kind == "sum":
                return f"{self.tensor_view} += {print_ir(self.value, print_out=False)}"
            elif self.operation.kind == "max":
                return f"{self.tensor_view} = max({print_ir(self.tensor_view.as_expr(), print_out=False)}, {print_ir(self.value, print_out=False)})"
            elif self.operation.kind == "min":
                return f"{self.tensor_view} = min({print_ir(self.tensor_view.as_expr(), print_out=False)}, {print_ir(self.value, print_out=False)})"
            else:
                raise RuntimeError(f"Unkonwn ReduceOp: {self.operation.kind}")
        else:
            return f"{self.tensor_view} = {print_ir(self.value, print_out=False)}"


class Loop(object):
    loop_cache = {}

    def __init__(self, r: Union[int, Union[List[Union[int, ConstInt]], range]], name="l", iter_kind=IterTypeKind.Spatial):
        if isinstance(r, (int, ir.Expr)):
            self.dom = Range(0, r, 1)
        elif isinstance(r, (list, tuple)):
            assert len(r) == 2 or len(r) == 3
            if len(r) == 2:
                self.dom = Range(r[0], r[1]-r[0], 1)
            elif len(r) == 3:
                self.dom = Range(r[0], (r[1]-r[0])//r[2], r[2])
        elif isinstance(r, range):
            start = 0 if r.start is None else r.start
            step = 1 if r.step is None else r.step
            assert r.stop is not None
            self.dom = Range(start, (r.stop-start)//step, step)
        else:
            raise ValueError(
                f"Can't use {r} of type {type(r)} to initialize Loop")

        self.name = NameGenerator.gen_name(name)
        self.iter_kind = iter_kind
        self.var = Var("int32", self.name)
        self.loop_cache[self.var] = self
        self.iterator = Iterator(self.var, self.dom, iter_kind)
        self.beg = self.dom.beg
        self.extent = self.dom.extent
        self.step = self.dom.step

    def iter_type(self):
        return self.iter_kind

    def __add__(self, other: Any):
        return self.var + other

    def __radd__(self, other):
        return other + self.var

    def __sub__(self, other):
        return self.var + other

    def __mul__(self, other):
        return self.var * other

    def __rmul__(self, other):
        return other * self.var

    def __truediv__(self, other):
        return self.var / other

    def __floordiv__(self, other):
        return self.var // other

    def __mod__(self, other):
        return self.var % other

    def __pow__(self, other):
        return self.var ** other

    def __rshift__(self, other):
        return self.var >> other

    def __lshift__(self, other):
        return self.var << other

    def __and__(self, other):
        return self.var & other

    def __or__(self, other):
        return self.var | other

    def __xor__(self, other):
        return self.var ^ other

    def __lt__(self, other):
        return self.var < other

    def __gt__(self, other):
        return self.var > other

    def __le__(self, other):
        return self.var <= other

    def __ge__(self, other):
        return self.var >= other

    def __eq__(self, other):
        return self.var == other

    def __ne__(self, other):
        return self.var != other

    def __neg__(self):
        return - self.var

    def __pos__(self):
        return self.var

    def __invert__(self):
        return ~self.var

    def __str__(self):
        return f"{str(self.iter_kind).split('.')[1]}Loop({self.name}, {print_ir(self.dom, print_out=False)})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.var)


class SpatialLoop(Loop):
    def __init__(self, r: Union[int, Union[List[Union[int, ConstInt]], range]], name="s"):
        super(SpatialLoop, self).__init__(r, name, IterTypeKind.Spatial)


SLoop = SpatialLoop


class ReduceLoop(Loop):
    def __init__(self, r: Union[int, Union[List[Union[int, ConstInt]], range]], name="r"):
        super(ReduceLoop, self).__init__(r, name, IterTypeKind.Reduce)


RLoop = ReduceLoop


class TensorizedLoop(Loop):
    def __init__(self, r: Union[int, Union[List[Union[int, ConstInt]], range]], name="t"):
        super(TensorizedLoop, self).__init__(r, name, IterTypeKind.Tensorized)


TLoop = TensorizedLoop


def make_prod_consum_graph(tensor):
    return ProdConsumGraph(tensor)


def max(a, b):
    if isinstance(a, TensorView):
        a = a.as_expr()
    if isinstance(b, TensorView):
        b = b.as_expr()

    if isinstance(a, ir.Expr) or isinstance(b, ir.Expr):
        return Max(a, b)


def pack_value(dtype, values):
    dtype = DType.make(dtype)
    sum_bits = reduce(lambda x, y: x + y, [v.dtype.bits for v in values], 0)
    if sum_bits != dtype.bits:
        raise ValueError(
            "pack_value requires the total bits of each value equals to the given dtype bits.")
    return PackValue(dtype, values)


def clip(value, lower, upper):
    lower = _to_expr(lower)
    upper = _to_expr(upper)
    return Max(lower, Min(upper, value))


def sqrt(value):
    value = _to_expr(value)
    return Call(value.dtype, "sqrt", [value])


def exp(value):
    if isinstance(value, TensorView):
        value = value.as_expr()
    elif isinstance(value, Loop):
        value = value.var
    else:
        value = _to_expr(value)
    return Call(value.dtype, "exp", [value])


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


const = make_const


def cast(dtype, value):
    dtype = DType.make(dtype)
    return Cast(dtype, value)
