from dominoc import ir as cir
from .arch import *
from .block import *
from .scalar_expr import *
from .stmt import *
from .kernel import *
from .dsl import *
from .functional import print_ir
from ..codegen import *
from ..type_system.dtype import DType
from typing import Optional, Union, List, Any, Dict
from functools import reduce
from .simplify import substitute_block, simplify, substitute_ir
from ..passes import flatten_array_access
from ..dse import MultiDimSpace, CategoricalSpace, CallablePolicy, spaces
from ..analysis import generate_fusion_plans

__all__ = [
    "program_lower",
    "program_build",
    "IRBuilderContext"
]


class IRBuilderContext(object):
    def __init__(self):
        # states
        self.tree = TreeContainer(None)
        self.stack = [self.tree]
        self.array_map = {}
        self.input_tensor_map = {}
        self.loop_relations = {}
        self.spatial_loop_set = {}
        # other control flags
        self.lower_to_tiles = False
        # tuning
        self.is_tuning_mode = False
        self.space = MultiDimSpace()
        self.config = None

    def enable_tuning(self, logfile="tmp_tuning_log.log"):
        self.is_tuning_mode = True
        if self.space.get_history_file() != logfile:
            self.space.set_history_file(logfile)

    def disable_tuning(self):
        self.is_tuning_mode = False

    def clear_state(self):
        self.tree = TreeContainer(None)
        self.stack = [self.tree]
        self.array_map = {}
        self.input_tensor_map = {}
        self.loop_relations = {}
        self.spatial_loop_set = {}

    def bind_array(self, var: Var, array: Array):
        assert isinstance(var, Var)
        assert isinstance(array, Array)
        self.array_map[var] = array

    def bind_input(self, var: Var, tensor: Tensor):
        assert isinstance(var, Var)
        assert isinstance(tensor, Tensor)
        self.input_tensor_map[var] = tensor

    def Array(self, shape, name="T", dtype="float32"):
        name = NameScope.current.gen_name(name)
        var = Var(dtype, name)
        array = Array(
            self, var, shape
        )
        self.bind_array(var, array)
        return array

    ## =---------------------------------------------------=##
    ## =         Implementation of primitives              =##
    ## =---------------------------------------------------=##
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

    def unroll_for(self, names=None, ranges=None, bindings=None):
        for_ctx = ForBlockContext(
            self, IterTypeKind.Unroll, names=names, ranges=ranges, bindings=bindings)
        return for_ctx

    def zigzag_for(self, names=None, ranges=None, bindings=None):
        for_ctx = ForBlockContext(
            self, IterTypeKind.Zigzag, names=names, ranges=ranges, bindings=bindings)
        return for_ctx

    def tile(self, mem_level: str, loops: List[Loop], annotation="Temporal"):
        assert isinstance(mem_level, str)
        for l in loops:
            assert isinstance(l, Loop)
        assert annotation in ["Spatial", "Temporal", "spatial", "temporal"]
        tile_ctx = TileBlockContext(
            self, mem_level, loops, annotation
        )
        return tile_ctx

    def map_var(self, name: str, expr: Expr):
        v = Var(expr.dtype, name)
        m = MapVar(v, expr)
        remap_ctx = ReMapBlockContext(
            self, [m]
        )
        return v

    def fill(self, array, value):
        raise NotImplementedError()

    # def init(self, tensor_view: TensorView, value: Expr):
    #     init_ctx = InitBlockContext(self, tensor_view, value)

    def split(self, loop: Loop, nparts=None, factors=None, rename=True):
        if factors is None:
            raise NotImplementedError(
                "Support for auto-tiling will be added later.")
        if nparts is not None:
            assert len(factors) == nparts
        new_loops = [
            Loop(f, name=loop.name, iter_kind=loop.iter_kind, rename=rename)
            for f in factors
        ]
        self.loop_relations[loop] = SplitRelation(new_loops)
        return new_loops

    def define_split(self, loop: Loop, nparts: int, constraints=None):
        if self.space.has_subspace(loop.name):
            return
        try:
            extent = loop.extent.value
            subspace = spaces.DimSplitSpace(
                extent, nparts, constraints=constraints)
            self.space.add_subspace(loop.name, subspace)
        except Exception as e:
            raise ValueError(f"Only support static shape\n{e}")

    def get_split(self, loop: Loop, policy: CallablePolicy):
        assert self.space.has_subspace(loop.name)
        subspace = self.space.get_subspace(loop.name)
        policy.set_inference_mode(not self.is_tuning_mode)
        return subspace.get_next(policy)

    def define_fuse(self, final_tensor: Tensor, levels: int):
        if self.space.has_subspace("fuse"):
            return
        plans = generate_fusion_plans(final_tensor, 2 * levels-2)
        print(f"Totally {len(plans)} fusion plans")
        self.space.add_subspace("fuse", CategoricalSpace(plans))

    def get_fuse(self, policy: CallablePolicy):
        policy.set_inference_mode(not self.is_tuning_mode)
        return self.space.get_subspace("fuse").get_next(policy)

    # def spatial(self, loop: Loop, dim="x"):
    #     self.spatial_loop_set[loop] = dim

    def sequential(self):
        scope_ctx = ScopeBlockContext(self, "Sequential")
        return scope_ctx

    def sharing(self):
        scope_ctx = ScopeBlockContext(self, "Sharing")
        return scope_ctx

    def pipeline(self):
        scope_ctx = ScopeBlockContext(self, "Pipeline")
        return scope_ctx

    def parallel(self):
        scope_ctx = ScopeBlockContext(self, "Parallel")
        return scope_ctx

    def load(self, target, lambda_func):
        if isinstance(target, Array):
            assert len(lambda_func.__code__.co_varnames) == len(target.shape)
            shape = []
            for s in target.shape:
                assert isinstance(s, (int, ConstInt))
                if isinstance(s, ConstInt):
                    shape.append(s.value)
                else:
                    shape.append(s)
            raise NotImplementedError()
        else:
            assert len(lambda_func.__code__.co_varnames) == 0

    def store(self, source, lambda_func):
        if isinstance(source, Array):
            raise NotImplementedError()
        else:
            assert len(lambda_func.__code__.co_varnames) == 0
            load = lambda_func()
            stmt = NdStore(load.mem_ref, load.indices, source)
            store_ctx = StmtBlockContext(self, stmt)

    def call(self, dtype: str, func_name: str, args: List[Expr]):
        stmt = Evaluate(Call(dtype, func_name, args))
        call_ctx = StmtBlockContext(self, stmt)

    def build_on_for_block(self, sub_trees, cur):
        if len(sub_trees) > 0:
            body = sub_trees[-1]
            for i in range(len(sub_trees) - 1):
                body = SeqBlock(
                    sub_trees[len(sub_trees) - i - 2], body)
        else:
            body = Evaluate(0)
        num_loops = len(cur.ctx.var_list)
        if cur.ctx.iter_type == IterTypeKind.Spatial or cur.ctx.iter_type == IterTypeKind.Reduce:
            for i in range(num_loops):
                body = ForBlock(
                    Iterator(cur.ctx.var_list[i], cur.ctx.ranges[i], cur.ctx.iter_type), body, cur.ctx.bindings[i])
        elif cur.ctx.iter_type == IterTypeKind.Unroll:
            for i in range(num_loops):
                if not (cur.ctx.ranges[i].extent.is_const() and cur.ctx.ranges[i].step.is_const()):
                    raise RuntimeError("Can't unroll dynamic loops")
                extent = cur.ctx.ranges[i].extent if isinstance(
                    cur.ctx.ranges[i].extent, int) else cur.ctx.ranges[i].extent.value
                step = cur.ctx.ranges[i].step if isinstance(
                    cur.ctx.ranges[i].step, int) else cur.ctx.ranges[i].step.value
                loop_var = cur.ctx.var_list[i]
                bodies = []
                for it in range(0, extent, step):
                    change_map = {loop_var: Add(
                        cur.ctx.ranges[i].beg, ConstInt(it))}
                    b = substitute_block(body, change_map)
                    bodies.append(b)
                assert len(bodies) > 0
                body = bodies[-1]
                for i in range(len(bodies) - 1):
                    body = SeqBlock(bodies[len(bodies) - i - 2], body)
        else:
            raise NotImplementedError()
        return body

    def build_on_tile_subtrees(self, sub_trees, cur):
        lift = False
        for part in sub_trees:
            if self.lower_to_tiles and isinstance(part, Arch):
                # all the sub-parts should be lifted to Arch
                lift = True
                break
        if lift and len(sub_trees) > 1:
            # all the parts should be in ctx.tile
            max_level = 0
            for part in sub_trees:
                if not isinstance(part, Arch):
                    raise RuntimeError(
                        "Not all statements are in the scope of ctx.tile!")
                if isinstance(part, ComputeLevel):
                    max_level = max(max_level, part.compute_level.value)
                if isinstance(part, MemoryLevel):
                    max_level = max(max_level, part.memory_level.value)

            # add the missing scope
            body = MemoryLevel(max_level, AtomBlock(
                Evaluate(0)), list(sub_trees))
            if isinstance(cur.ctx, ScopeBlockContext):
                body.set_scope(cur.ctx.scope)
            else:
                body.set_scope("Sequential")
        else:
            # all the parts are not Arch nodes
            body = sub_trees[-1]
            for i in range(len(sub_trees) - 1):
                body = SeqBlock(
                    sub_trees[len(sub_trees) - i - 2], body)
        return body

    def build_on_tile_block(self, sub_trees, cur):
        if len(sub_trees) > 0:
            body = self.build_on_tile_subtrees(sub_trees, cur)
        else:
            body = Evaluate(0)
        num_loops = len(cur.ctx.loops)
        # = below is code for lowering to memory-tree
        if self.lower_to_tiles:
            iterators = []
            assert isinstance(cur.ctx.mem_level,
                              str) and cur.ctx.mem_level[0] == 'L'
            level = int(cur.ctx.mem_level[1:])
            if (num_loops == 0):
                if isinstance(body, Arch):
                    # already formed as tile, must be memory level
                    body = MemoryLevel(level, _to_block(
                        ExprList(iterators)), [body])
                    body.set_annotation("Temporal")
                else:
                    # make a compute level first
                    body = ComputeLevel(level, _to_block(body), [])
                    body = MemoryLevel(level, _to_block(
                        ExprList(iterators)), [body])
                    body.set_annotation("Temporal")
            else:
                tail = num_loops - 1
                last_type = "Spatial" if cur.ctx.loops[tail] in self.spatial_loop_set else "Temporal"
                while tail >= 0:
                    cur_type = "Spatial" if cur.ctx.loops[tail] in self.spatial_loop_set else "Temporal"
                    if cur_type == last_type:
                        iterators = [cur.ctx.loops[tail].iterator] + iterators
                    else:
                        raise RuntimeError(
                            "Please don't specify loop annotation outside tile.")
                        # if isinstance(body, Arch):
                        #     # already formed as tile, must be memory level
                        #     body = MemoryLevel(level, _to_block(
                        #         ExprList(iterators)), [body])
                        #     body.set_annotation(last_type)
                        # else:
                        #     # make a compute level first
                        #     body = ComputeLevel(level, _to_block(body), [])
                        #     body = MemoryLevel(level, _to_block(
                        #         ExprList(iterators)), [body])
                        #     body.set_annotation(last_type)
                        # iterators = [cur.ctx.loops[tail].iterator]
                        # last_type = cur_type
                    tail -= 1
                if len(iterators):
                    if isinstance(body, Arch):
                        # already formed as tile, must be memory level
                        body = MemoryLevel(level, _to_block(
                            ExprList(iterators)), [body])
                        body.set_annotation(cur.ctx.annotation)
                    else:
                        # make a compute level first
                        body = ComputeLevel(level, _to_block(body), [])
                        body = MemoryLevel(level, _to_block(
                            ExprList(iterators)), [body])
                        body.set_annotation(cur.ctx.annotation)
        # = below is code for lowering to loops =#
        else:
            for i in range(num_loops):
                idx = num_loops - i - 1
                loop = cur.ctx.loops[idx]
                if loop in self.spatial_loop_set:
                    binding = "Spatial"
                else:
                    binding = "Temporal"
                body = ForBlock(
                    loop.iterator, body, binding
                )
            body = AttrBlock(
                "tile", Var("int32"), cur.ctx.mem_level, body
            )
        return body

    def build_on_scope_block(self, sub_trees, cur):
        if len(sub_trees) > 0:
            body = self.build_on_tile_subtrees(sub_trees, cur)
        else:
            body = Evaluate(0)
        if not self.lower_to_tiles:
            body = AttrBlock(
                "scope", Var("int32"), cur.ctx.scope, body
            )
        return body

    def build_on_alloc_block(self, sub_trees, cur):
        if len(sub_trees) > 0:
            body = sub_trees[-1]
            for i in range(len(sub_trees) - 1):
                body = SeqBlock(
                    sub_trees[len(sub_trees) - i - 2], body)
        else:
            body = Evaluate(0)
        body = NdAllocBlock(
            cur.ctx.array.var, cur.ctx.array.shape, cur.ctx.array.scope, body)
        return body

    def build_on_stmt_block(self, sub_trees, cur):
        assert len(sub_trees) == 0
        if self.lower_to_tiles:
            assert isinstance(cur.ctx.stmt, NdStore)
            stmt_level = ComputeLevel(0, _to_block(cur.ctx.stmt), [])
            stmt_level.set_produce_var(cur.ctx.stmt.mem_ref.var)
            return stmt_level
        else:
            return cur.ctx.stmt
        # if len(sub_trees) == 1:
        #     body = sub_trees[0]
        #     body = SeqBlock(cur.ctx.stmt, body)
        #     return body
        # else:
        #     body = cur.ctx.stmt
        #     return body

    def build_on_remap_block(self, sub_trees, cur):
        if len(sub_trees) > 0:
            body = sub_trees[-1]
            for i in range(len(sub_trees) - 1):
                body = SeqBlock(
                    sub_trees[len(sub_trees) - i - 2], body)
        else:
            body = Evaluate(0)
        body = ReMapBlock(
            cur.ctx.map_vars, body)
        return body

    ## =---------------------------------------------------=##
    ## =                  Build Program                    =##
    ## =---------------------------------------------------=##
    def build(self):
        def builder(cur):
            sub_trees = [builder(x) for x in cur.children]
            if cur.is_root():
                if len(sub_trees) == 1:
                    return sub_trees[0]
                elif len(sub_trees) > 1:
                    if self.lower_to_tiles:
                        return MemoryLevel(114514, AtomBlock(Evaluate(0)), sub_trees)
                    else:
                        body = sub_trees[-1]
                        for i in range(len(sub_trees) - 1):
                            body = SeqBlock(
                                sub_trees[len(sub_trees) - i - 2], body)
                        return body
                else:
                    return AtomBlock(Evaluate(0))
            elif isinstance(cur.ctx, ForBlockContext):
                return self.build_on_for_block(sub_trees, cur)
            elif isinstance(cur.ctx, TileBlockContext):
                return self.build_on_tile_block(sub_trees, cur)
            elif isinstance(cur.ctx, ScopeBlockContext):
                return self.build_on_scope_block(sub_trees, cur)
            elif isinstance(cur.ctx, AllocBlockContext):
                return self.build_on_alloc_block(sub_trees, cur)
            elif isinstance(cur.ctx, StmtBlockContext):
                return self.build_on_stmt_block(sub_trees, cur)
            elif isinstance(cur.ctx, ReMapBlockContext):
                return self.build_on_remap_block(sub_trees, cur)
            else:
                raise NotImplementedError(f"{cur.ctx}")
        ret = builder(self.tree)

        # substitute the loops
        smap = {}
        for l, rel in self.loop_relations.items():
            if isinstance(rel, SplitRelation):
                loops = rel.sub_loops
                assert len(loops) > 0
                strides = [1]
                for s in reversed(loops[1:]):
                    strides = [Mul(strides[0], s.extent)] + strides
                flattened = loops[0] * strides[0]
                for ll, ss in zip(loops[1:], strides[1:]):
                    flattened += Mul(ll.var, ss)
                smap[l.var] = flattened
        if not self.lower_to_tiles:
            ret = substitute_block(ret, smap)

        return ret


def flatten_arrays(body: Block, arrays: List[Array]):
    var_list = []
    strides = []
    for array in arrays:
        var_list.append(array.var)
        strides.append(ExprList([_to_expr(x) for x in array.shape]))
    return flatten_array_access(body, var_list, strides)


def program_lower(func, tensor_inputs: List[Tensor], scalar_inputs=None, ctx=None):
    ctx = IRBuilderContext() if ctx is None else ctx
    assert isinstance(ctx, IRBuilderContext)

    tensor_input_vars = [t.var for t in tensor_inputs]

    input_arrays = []
    for v, t in zip(tensor_input_vars, tensor_inputs):
        ctx.bind_input(v, t)
        # [reduce(lambda x, y: x * y, t.shape, 1)]
        array = Array(ctx, v, t.shape)
        input_arrays.append(array)
        ctx.bind_array(v, array)

    scalar_inputs = [] if scalar_inputs is None else scalar_inputs

    func(ctx, *input_arrays, *scalar_inputs)
    body = ctx.build()

    # basic passes (TODO: pass management still in progress)
    body = simplify(body)
    body = flatten_arrays(body, input_arrays)

    signature = KernelSignature(
        func.__name__, tensor_input_vars, scalar_inputs)
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

    if target == "c":
        code = codegen_c(kernel.body)
    elif target == "arm_m":
        code = codegen_arm_m(kernel.body)
    else:
        raise NotImplementedError()

    kernel.source = code

    return kernel
