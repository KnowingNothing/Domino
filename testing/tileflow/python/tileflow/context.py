from dominoc import ir as cir
import domino.program_ir as dir
import domino.codegen as dgen
from typing import List, Dict

__all__ = ["register_workload", "Context", "arch_lower", "arch_build"]


WORKLOAD_MAP = {}

def register_workload(func):
    name = func.__name__
    assert name not in WORKLOAD_MAP
    WORKLOAD_MAP[name] = func
    return func


class Context(dir.IRBuilderContext):
    def __init__(self):
        super().__init__()
        self.lower_to_tiles = True
    
    def set_space(self, space):
        self.space = space
        
    def set_target_tileflow(self):
        self.lower_to_tiles = True
        
    def split(self, loop, nparts=None, factors=None):
        return super().split(loop, nparts=nparts, factors=factors, rename=False)


def remap_tiled_loops(body: dir.Arch, loop_relations: Dict[dir.Var, dir.Var]):
    """
    This is used to simplify the code generation for TileFlow
    Because TileFlow requires all the tiled loops use the same name
    """
    return dir.substitute_ir(body, loop_relations)


def arch_lower(ctx):
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
