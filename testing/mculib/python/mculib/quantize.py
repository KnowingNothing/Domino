from domino.program_ir import *


def requantize(ctx, acc_v, scale_v, offset_v, clip_min, clip_max):
    # tmp = ctx.map_var(f"tmp_{acc_v.id.value}", cast(
    #     "int32", (cast("float32", acc_v) * scale_v)) + offset_v)
    tmp = cast("int32", (cast("float32", acc_v) * scale_v)) + offset_v
    return cast("int8", clip(tmp, clip_min, clip_max))