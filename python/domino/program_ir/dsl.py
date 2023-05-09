from dominoc import ir
from ..type_system import DType
from .scalar_expr import *
from .stmt import *
from .functional import print_ir
from .block import _to_block
from ..passes import (get_input_tensor_vars, ProdConsumGraph,
                      get_input_tensor_indices)
from typing import List, Union, Any, Optional
from functools import reduce


__all__ = [
    "NameGenerator",
    "NameScope",
    "ReduceOp",
    "ElemOp",
    "Tensor",
    "TensorView",
    "ConstTensor",
    "Loop",
    "SpatialLoop",
    "ReduceLoop",
    "TensorizedLoop",
    "SLoop",
    "RLoop",
    "TLoop",
    "make_prod_consum_graph",
    "max",
    "cast",
    "pack_value",
    "clip",
    "sqrt",
    "exp",
    "make_const",
    "const",
    "SyntaxContext",
    "BlockContext",
    "ForBlockContext",
    "StmtBlockContext",
    "TileBlockContext",
    "AllocBlockContext",
    "ReMapBlockContext",
    "ScopeBlockContext",
    "TreeContainer",
    "LoopRelation",
    "SplitRelation",
    "Array"]


class NameGenerator(object):
    def __init__(self, only_capital=False):
        self.name_cache = {}
        self.only_capital = only_capital
        self.capital_used = set()
        self.choices = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def gen_name(self, hint: str):
        if self.only_capital:
            if hint not in self.name_cache:
                if hint in self.choices and hint not in self.capital_used:
                    self.capital_used.add(hint)
                    self.name_cache[hint] = hint
                    return hint
            for c in self.choices:
                if c not in self.capital_used:
                    self.capital_used.add(c)
                    self.name_cache[hint] = c
                    return c
            raise RuntimeError("No enough characters to use.")
        else:
            parts = hint.split("_")
            if parts[0] not in self.name_cache:
                self.name_cache[parts[0]] = 0
                return parts[0]
            else:
                v = self.name_cache[parts[0]]
                self.name_cache[parts[0]] += 1
                name = parts[0] + f"_{v}"
                return name

class DefaultNameScope:
    def __init__(self):
        self.gen = NameGenerator()
        
    def gen_name(self, hint: str):
        return self.gen.gen_name(hint)
        
class NameScope:
    current = DefaultNameScope()
    def __init__(self, only_capital=False):
        self.pre = NameScope.current
        self.gen = NameGenerator(only_capital)
        NameScope.current = self
        
    def gen_name(self, hint: str):
        return self.gen.gen_name(hint)
        
    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is None:
            cur_scope = NameScope.current
            NameScope.current = cur_scope.pre
            return True
        else:
            raise RuntimeError(exc_value)


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
            dtype: Union[DType, str] = "float32",
            ctx = None):
        self.shape = shape
        self.name = NameScope.current.gen_name(name)
        self.dtype = dtype
        self.var = Var(self.dtype, self.name)
        self.init: Optional[Compute] = None
        self.updates: List[Compute] = []
        # FIXME: how to avoid global tensor cache?
        self.tensor_cache[self.var] = self
        self.ctx = ctx

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
        # for ir_builder
        if self.ctx is not None:
            def store_func():
                return NdLoad(MemRef(self.var, 0), new_keys)
            self.ctx.store(value, store_func)

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
    def __init__(self, tensor_view: TensorView, value: Expr, operation: ComputeOp, reduce_axis: List[Var]):
        self.tensor_view = tensor_view
        assert isinstance(value, Expr)
        self.value = value
        self.operation = operation
        self.reduce_axis = reduce_axis
        
    def as_stmt(self):
        load = self.tensor_view.as_expr()
        return NdStore(load.mem_ref, load.indices, self.value)

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

    def __init__(self, r: Union[int, Union[List[Union[int, ConstInt]], range]], name="l", iter_kind=IterTypeKind.Hybrid, rename=True):
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
        if rename:
            self.name = NameScope.current.gen_name(name)
        else:
            self.name = name
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
        if isinstance(other, Loop):
            return self.var + other.var
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
    else:
        return a if a > b else b


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


class SyntaxContext(object):
    def __init__(self, ir_builder) -> None:
        self.ir_builder = ir_builder


class BlockContext(SyntaxContext):
    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError()


class Array(object):
    def __init__(self, ir_builder, var, shape, scope="global", origin: Optional["Array"] = None, slices=None):
        self.ir_builder = ir_builder
        self.var = var
        self.shape = shape
        self.scope = scope
        self.origin = origin
        self.slices = tuple(slice(0, s, 1)
                            for s in self.shape) if slices is None else tuple(slices)
        assert len(self.slices) == len(self.shape)

    @property
    def dtype(self):
        return self.var.dtype

    def ref(self, *indices):
        return ArrayRef(self.var, indices)

    def calculate_slice_indices(self, indices):
        # FIXME: don't support slices in indices
        if self.origin is None:
            return indices
        origin_indices = []
        counter = 0
        length = len(indices)
        for s in self.slices:
            if isinstance(s, slice):
                assert counter < length
                start = s.start if s.start is not None else 0
                step = s.step if s.step is not None else 1
                idx = start + indices[counter] * step
                counter += 1
                origin_indices.append(idx)
            else:
                origin_indices.append(s)
        return self.origin.calculate_slice_indices(origin_indices)

    def __str__(self):
        return f"Array({self.var.id.value}, {self.shape}, {self.var.dtype}, {self.scope})"

    def __repr__(self):
        return str(self)

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        assert len(keys) == len(self.shape), f"{keys} vs {self.shape}"
        new_shape = []
        indices = []
        for i, k in enumerate(keys):
            if isinstance(k, slice):
                start = 0 if k.start is None else k.start
                stop = self.shape[i] if k.stop is None else k.stop
                step = 1 if k.step is None else k.step
                # TODO: better index bound check
                start_value = start.value if isinstance(
                    start, ConstInt) else start
                stop_value = stop.value if isinstance(stop, ConstInt) else stop
                shape_value = self.shape[i].value if isinstance(
                    self.shape[i], ConstInt) else self.shape[i]
                if all(isinstance(x, int) for x in [start_value, stop_value, shape_value]):
                    assert start_value >= 0 and start_value < stop_value
                    assert stop_value < shape_value

                extent = (stop - start) // step
                new_shape.append(extent)
            else:
                if isinstance(k, Loop):
                    k = k.var
                # TODO: better index bound check
                value_k = k
                if isinstance(k, ConstInt):
                    value_k = k.value
                value_s = self.shape[i]
                if isinstance(self.shape[i], ConstInt):
                    value_s = self.shape[i].value
                if isinstance(value_k, int) and isinstance(value_s, int):
                    assert value_k < value_s and value_k >= 0

                indices.append(k)
        if len(new_shape):
            return Array(self.ir_builder, self.var, new_shape, scope=self.scope, origin=self, slices=keys)
        else:
            return NdLoad(MemRef(self.var, 0), self.calculate_slice_indices(indices))

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            keys = (keys,)
        assert len(keys) == len(self.shape)
        new_shape = []
        indices = []
        for i, k in enumerate(keys):
            if isinstance(k, slice):
                start = 0 if k.start is None else k.start
                stop = self.shape[i] if k.stop is None else k.stop
                step = 1 if k.step is None else k.step
                # TODO: better index bound check
                start_value = start.value if isinstance(
                    start, ConstInt) else start
                stop_value = stop.value if isinstance(stop, ConstInt) else stop
                shape_value = self.shape[i].value if isinstance(
                    self.shape[i], ConstInt) else self.shape[i]
                if all(isinstance(x, int) for x in [start_value, stop_value, shape_value]):
                    assert start_value >= 0 and start_value < stop_value
                    assert stop_value < shape_value

                extent = (stop - start) // step
                new_shape.append(extent)
            else:
                # TODO: better index bound check
                value_k = k
                if isinstance(k, ConstInt):
                    value_k = k.value
                value_s = self.shape[i]
                if isinstance(self.shape[i], ConstInt):
                    value_s = self.shape[i].value
                if isinstance(value_k, int) and isinstance(value_s, int):
                    assert value_k < value_s and value_k >= 0

                indices.append(k)
        if len(new_shape):
            raise NotImplementedError()
            return Array(self.ir_builder, self.var, new_shape, scope=self.scope, origin=self, slices=keys)
        else:
            def store_func(): return self.__getitem__(keys)
            self.ir_builder.store(value, store_func)
            # return NdStore(MemRef(self.var, 0), self.calculate_slice_indices(indices), value)


class AllocBlockContext(BlockContext):
    def __init__(self, ir_builder, shape, scope="global", dtype="float32", name=""):
        super(AllocBlockContext, self).__init__(ir_builder)
        var = Var(dtype, name)
        self.array = Array(ir_builder, var, shape, scope=scope)
        self.ir_builder.bind_array(var, self.array)
        node = TreeContainer(self)
        self.ir_builder.stack[-1].add_child(node)
        self.ir_builder.stack.append(node)


class ForBlockContext(BlockContext):
    def __init__(self, ir_builder, iter_type, names=None, ranges=None, bindings=None):
        super(ForBlockContext, self).__init__(ir_builder)
        self.iter_type = iter_type
        if names is None:
            raise ValueError("Please initialize for loops with names")
        if ranges is None:
            raise ValueError("Please initialize for loops with ranges")
        names = [names] if not isinstance(names, (list, tuple)) else names
        ranges = [ranges] if not isinstance(ranges, (list, tuple)) else ranges

        self.names = []
        for name in names:
            if isinstance(name, str):
                self.names.append(name)
            elif isinstance(name, ConstString):
                self.names.append(name.value)
            else:
                raise ValueError(
                    "Please use string or ConstString type for loop names.")

        if ranges is None:
            raise ValueError("Please initialize for loops with ranges")
        self.ranges = []
        for r in ranges:
            if isinstance(r, range):
                start = 0 if r.start is None else r.start
                stop = r.stop
                step = 1 if r.step is None else r.step
                self.ranges.append(Range(start, stop - start, step))
            elif isinstance(r, Range):
                self.ranges.append(r)
            else:
                raise ValueError(
                    "Please use range or Range type for loop ranges.")

        if len(names) != len(ranges):
            raise ValueError(
                "Please provide the same number of loop names and ranges.")
        if bindings is None:
            self.bindings = [ConstString("") for _ in names]
        else:
            self.bindings = []
            for b in bindings:
                if isinstance(b, str):
                    self.bindings.append(ConstString(b))
                elif isinstance(b, ConstString):
                    self.bindings.append(b)
                else:
                    raise ValueError(
                        "Please use string or ConstString type for loop bindings.")

        self.var_list = [Var("int32", name) for name in self.names]

    def __enter__(self):
        node = TreeContainer(self)
        self.ir_builder.stack[-1].add_child(node)
        self.ir_builder.stack.append(node)

        ret = self.var_list
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is None:
            while len(self.ir_builder.stack) > 0 and self.ir_builder.stack[-1].ctx != self:
                self.ir_builder.stack.pop()
            if self.ir_builder.stack[-1].ctx == self:
                self.ir_builder.stack.pop()
            return True
        else:
            raise RuntimeError(exc_value)


class TileBlockContext(BlockContext):
    def __init__(self, ir_builder, mem_level: str, loops: List[Loop], annotation: str):
        super(TileBlockContext, self).__init__(ir_builder)
        self.mem_level = mem_level
        self.loops = loops
        self.annotation = annotation
        
    def merge_loops(self, other: "TileBlockContext"):
        assert self.mem_level == other.mem_level
        assert self.annotation == other.annotation
        visit = set()
        for l in self.loops:
            visit.add(l)
        for l in other.loops:
            if l not in visit:
                self.loops.append(l)
                visit.add(l)

    def __enter__(self):
        node = TreeContainer(self)
        self.ir_builder.stack[-1].add_child(node)
        self.ir_builder.stack.append(node)

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is None:
            while len(self.ir_builder.stack) > 0 and self.ir_builder.stack[-1].ctx != self:
                self.ir_builder.stack.pop()
            if self.ir_builder.stack[-1].ctx == self:
                self.ir_builder.stack.pop()
            return True
        else:
            raise RuntimeError(exc_value)


class ScopeBlockContext(BlockContext):
    def __init__(self, ir_builder, scope: str):
        super(ScopeBlockContext, self).__init__(ir_builder)
        self.scope = scope

    def __enter__(self):
        node = TreeContainer(self)
        self.ir_builder.stack[-1].add_child(node)
        self.ir_builder.stack.append(node)

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is None:
            while len(self.ir_builder.stack) > 0 and self.ir_builder.stack[-1].ctx != self:
                self.ir_builder.stack.pop()
            if self.ir_builder.stack[-1].ctx == self:
                self.ir_builder.stack.pop()
            return True
        else:
            raise RuntimeError(exc_value)


# class InitBlockContext(BlockContext):
#     def __init__(self, ir_builder, tensor_view: TensorView, value: Expr):
#         super(InitBlockContext, self).__init__(ir_builder)
#         self.tensor_view = tensor_view
#         self.value = value
#         node = TreeContainer(self)
#         self.ir_builder.stack[-1].add_child(node)


class StmtBlockContext(BlockContext):
    def __init__(self, ir_builder, stmt):
        super(StmtBlockContext, self).__init__(ir_builder)
        self.stmt = stmt
        node = TreeContainer(self)
        self.ir_builder.stack[-1].add_child(node)


class ReMapBlockContext(BlockContext):
    def __init__(self, ir_builder, map_vars: List[MapVar]):
        super(ReMapBlockContext, self).__init__(ir_builder)
        for m in map_vars:
            assert isinstance(m, MapVar)
        self.map_vars = map_vars
        node = TreeContainer(self)
        self.ir_builder.stack[-1].add_child(node)
        self.ir_builder.stack.append(node)


class TreeContainer(object):
    def __init__(self, ctx):
        assert ctx is None or isinstance(ctx, SyntaxContext)
        self.ctx = ctx
        self.children = []

    def add_child(self, child):
        assert isinstance(child, TreeContainer)
        self.children.append(child)
        
    def add_front_child(self, child):
        assert isinstance(child, TreeContainer)
        self.children = [child] + self.children

    def is_root(self):
        return self.ctx is None
    
    def walk(self, callback):
        callback(self)
        for c in self.children:
            c.walk(callback)
            
    def post_order_walk(self, callback):
        for c in self.children:
            c.post_order_walk(callback)
        callback(self)


class LoopRelation(object):
    pass


class SplitRelation(LoopRelation):
    def __init__(self, sub_loops) -> None:
        self.sub_loops = sub_loops
