from .block import *
from .scalar_expr import *
from .stmt import *
from .kernel import *
from .dsl import *
from .functional import print_ir
from ..codegen import codegen_c
from typing import Optional

__all__ = [
    "program_lower",
    "program_build"
]


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

    def calculate_slice_indices(self, indices):
        if self.origin is None:
            return indices
        origin_indices = []
        counter = 0
        length = len(indices)
        for s in self.slices:
            if isinstance(s, slice):
                assert counter < length
                idx = s.start + indices[counter] * s.step
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


class StmtBlockContext(BlockContext):
    def __init__(self, ir_builder, stmt):
        super(StmtBlockContext, self).__init__(ir_builder)
        self.stmt = stmt
        node = TreeContainer(self)
        self.ir_builder.stack[-1].add_child(node)
        # self.stack.append(node)


class TreeContainer(object):
    def __init__(self, ctx):
        self.ctx = ctx
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def is_root(self):
        return self.ctx is None


class IRBuilderContext(object):
    def __init__(self):
        self.tree = TreeContainer(None)
        self.stack = [self.tree]
        self.array_map = {}
        self.input_tensor_map = {}

    def bind_array(self, var: Var, array: Array):
        assert isinstance(var, Var)
        assert isinstance(array, Array)
        self.array_map[var] = array

    def bind_input(self, var: Var, tensor: Tensor):
        assert isinstance(var, Var)
        assert isinstance(tensor, Tensor)
        self.input_tensor_map[var] = tensor

    def alloc(self, shape, scope="global", dtype="float32", name=""):
        alloc_ctx = AllocBlockContext(
            self, shape, scope=scope, dtype=dtype, name=name)
        return alloc_ctx.array

    def spatial_for(self, names=None, ranges=None, bindings=None):
        for_ctx = ForBlockContext(
            self, IterTypeKind.Spatial, names=names, ranges=ranges, bindings=bindings)
        return for_ctx

    def reduce_for(self, names=None, ranges=None, bindings=None):
        for_ctx = ForBlockContext(
            self, IterTypeKind.Reduce, names=names, ranges=ranges, bindings=bindings)
        return for_ctx

    def zigzag_for(self, names=None, ranges=None, bindings=None):
        raise NotImplementedError()

    def fill(self, array, value):
        raise NotImplementedError()

    def load(self, target, lambda_func):
        raise NotImplementedError()

    def store(self, source, lambda_func):
        if isinstance(source, Array):
            raise NotImplementedError()
        else:
            assert len(lambda_func.__code__.co_varnames) == 0
            load = lambda_func()
            stmt = NdStore(load.mem_ref, load.indices, source)
            store_ctx = StmtBlockContext(self, stmt)

    def mma(self, output=None, input_a=None, input_b=None, input_c=None, layout_a=None, layout_b=None):
        raise NotImplementedError()

    def build(self):
        def builder(cur):
            sub_trees = [builder(x) for x in cur.children]
            if cur.is_root():
                assert len(sub_trees) == 1
                return sub_trees[0]
            elif isinstance(cur.ctx, ForBlockContext):
                assert len(sub_trees) >= 1, len(sub_trees)
                body = sub_trees[-1]
                for i in range(len(sub_trees) - 1):
                    body = SeqBlock(sub_trees[len(sub_trees) - i - 2], body)
                num_loops = len(cur.ctx.var_list)
                for i in range(num_loops):
                    body = ForBlock(
                        Iterator(cur.ctx.var_list[i], cur.ctx.ranges[i], cur.ctx.iter_type), body, cur.ctx.bindings[i])
                return body
            elif isinstance(cur.ctx, AllocBlockContext):
                assert len(sub_trees) >= 1
                body = sub_trees[-1]
                for i in range(len(sub_trees) - 1):
                    body = SeqBlock(sub_trees[len(sub_trees) - i - 2], body)
                body = NdAllocBlock(
                    cur.ctx.array.var, cur.ctx.array.shape, cur.ctx.array.scope, body)
                return body
            elif isinstance(cur.ctx, StmtBlockContext):
                assert len(sub_trees) == 0
                return cur.ctx.stmt
                # if len(sub_trees) == 1:
                #     body = sub_trees[0]
                #     body = SeqBlock(cur.ctx.stmt, body)
                #     return body
                # else:
                #     body = cur.ctx.stmt
                #     return body
            else:
                raise NotImplementedError()
        ret = builder(self.tree)
        return ret


def program_lower(func, tensor_inputs, scalar_inputs=None, ctx=None):
    ctx = IRBuilderContext() if ctx is None else ctx
    assert isinstance(ctx, IRBuilderContext)

    tensor_input_vars = [Var(t.dtype, t.name) for t in tensor_inputs]
    input_arrays = []
    for v, t in zip(tensor_input_vars, tensor_inputs):
        ctx.bind_input(v, t)
        array = Array(ctx, v, t.shape)
        input_arrays.append(array)
        ctx.bind_array(v, array)
    assert scalar_inputs is None, "Currently don't support scalar inputs"

    func(ctx, *input_arrays)
    body = ctx.build()

    signature = KernelSignature(func.__name__, tensor_input_vars)
    kernel = Kernel(signature, body)
    return kernel


def program_build(func, tensor_inputs=None, scalar_inputs=None, ctx=None, target="c"):
    if not isinstance(func, Kernel):
        assert tensor_inputs is not None
        ctx = IRBuilderContext() if ctx is None else ctx
        assert isinstance(ctx, IRBuilderContext)
        kernel = program_lower(func, tensor_inputs,
                       scalar_inputs=scalar_inputs, ctx=ctx)
    else:
        assert ctx is not None and isinstance(ctx, IRBuilderContext)
        kernel = func

    for k, v in ctx.array_map.items():
        print(k, v)
    
    if target == "c":
        code = codegen_c(kernel.body)
    else:
        raise NotImplementedError()
    
    kernel.source = code
    
    return kernel
