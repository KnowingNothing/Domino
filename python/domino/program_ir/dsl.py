from ..type_system import DType
from .scalar_expr import Expr, Var, ConstVar, ConstInt, Range, Iterator, IterTypeKind, _to_expr, NdLoad, MemRef
from .functional import print_ir
from typing import List, Union, Any


__all__ = ["Tensor", "ConstTensor", "Loop", "SpatialLoop",
           "ReduceLoop", "TensorizedLoop", "SLoop", "RLoop", "TLoop"]


class Tensor(object):
    name_cache = {}

    def __init__(
            self,
            shape: List[Union[int, Expr]],
            name: str = "",
            dtype: Union[DType, str] = "float32"):
        self.shape = shape
        self.name = Tensor._gen_name(name)
        self.dtype = dtype
        self.var = Var(self.dtype, self.name)

    def is_const(self):
        return False

    def __getitem__(self, keys):
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
        return TensorView(new_shape, self, new_keys)

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

    def __getitem__(self, keys):
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


class Loop(object):
    name_cache = {}

    def __init__(self, r: Union[int, Union[List[Union[int, ConstInt]], range]], name="", iter_kind=IterTypeKind.Spatial):
        if isinstance(r, int):
            self.dom = Range(0, r, 1)
        elif isinstance(r, (list, tuple)):
            assert len(r) == 2 or len(r) == 3
            if len(r) == 2:
                self.dom = Range(r[0], r[1], 1)
            elif len(r) == 3:
                self.dom = Range(r[0], r[1], r[2])
        elif isinstance(r, range):
            start = 0 if r.start is None else r.start
            step = 1 if r.step is None else r.step
            assert r.stop is not None
            self.dom = Range(start, r.stop, step)
        else:
            raise ValueError(
                f"Can't use {r} of type {type(r)} to initialize Loop")

        self.name = Loop._gen_name(name)
        self.iter_kind = iter_kind
        self.var = Var("int32", name)
        self.iterator = Iterator(self.var, self.dom, iter_kind)

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


class SpatialLoop(Loop):
    def __init__(self, r: Union[int, Union[List[Union[int, ConstInt]], range]], name=""):
        super(SpatialLoop, self).__init__(r, name, IterTypeKind.Spatial)


SLoop = SpatialLoop


class ReduceLoop(Loop):
    def __init__(self, r: Union[int, Union[List[Union[int, ConstInt]], range]], name=""):
        super(ReduceLoop, self).__init__(r, name, IterTypeKind.Reduce)


RLoop = ReduceLoop


class TensorizedLoop(Loop):
    def __init__(self, r: Union[int, Union[List[Union[int, ConstInt]], range]], name=""):
        super(TensorizedLoop, self).__init__(r, name, IterTypeKind.Tensorized)


TLoop = TensorizedLoop
