from dominoc import ir as cir
import domino.program_ir as dir
import domino.codegen as dgen
from typing import List, Dict

__all__ = ["Context", "arch_lower", "arch_build"]


class Context(dir.IRBuilderContext):
    def __init__(self):
        super().__init__()


def remap_tiled_loops(body: dir.Arch, loop_relations: Dict[dir.Var, dir.Var]):
    """
    This is used to simplify the code generation for TileFlow
    Because TileFlow requires all the tiled loops use the same name
    """
    return dir.substitute_ir(body, loop_relations)


def arch_lower(func, tensor_inputs: List[dir.Tensor], scalar_inputs=None, ctx=None, plan=None, final_tensor=None):
    if plan is not None:
        assert ctx is None, "Do not provide ctx for plan"
    ctx = Context() if ctx is None else ctx
    ctx.set_target_tileflow()
    assert isinstance(ctx, Context)

    tensor_input_vars = [t.var for t in tensor_inputs]

    if plan is None:
        input_arrays = []
        for v, t in zip(tensor_input_vars, tensor_inputs):
            ctx.bind_input(v, t)
            # [reduce(lambda x, y: x * y, t.shape, 1)]
            array = dir.Array(ctx, v, t.shape)
            input_arrays.append(array)
            ctx.bind_array(v, array)
        scalar_inputs = [] if scalar_inputs is None else scalar_inputs
        func(ctx, *input_arrays, *scalar_inputs)
    else:
        for t in tensor_inputs:
            t.ctx = ctx
        scalar_inputs = [] if scalar_inputs is None else scalar_inputs
        func(ctx, *tensor_inputs, *scalar_inputs)
        assert final_tensor is not None
        plan.apply(final_tensor, ctx)
    body = ctx.build()
    body = dir.simplify(body)

    reverse_relation = {}
    for k, v in ctx.loop_relations.items():
        for vv in v.sub_loops:
            reverse_relation[vv.var] = k.var
    body = remap_tiled_loops(body, reverse_relation)
    return body


def arch_build(func, tensor_inputs=None, scalar_inputs=None, ctx=None, target="tileflow"):
    if not isinstance(func, cir.Arch):
        assert tensor_inputs is not None
        ctx = Context() if ctx is None else ctx
        assert isinstance(ctx, Context)
        kernel = arch_lower(func, tensor_inputs,
                            scalar_inputs=scalar_inputs, ctx=ctx)
    else:
        kernel = func

    if target == "tileflow":
        code = dgen.codegen_tileflow(kernel)
    else:
        raise NotImplementedError()

    return code
