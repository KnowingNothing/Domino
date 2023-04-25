import domino.program_ir as dir
from ..ops import Conv2d, Conv2d_nchw


def conv_chain_dataflow(ctx, tI, tW1, tW2, batch, height, width, in_channel, out_channel_1, out_channel_2, levels, define_tiling_space=True, second_kernel_size=3):
    """
    tA: [batch, height, width, in_channel]
    tB: [out_channel_1, kh, kw, in_channel]
    tC: [out_channel_2, kh, kw, out_channel_1]
    """
    b, h, w, c, l, k, r, s, u, v = [dir.Loop(x, name=y) for (x, y) in zip(
        [batch, height, width, in_channel, out_channel_1, out_channel_2, 3, 3, second_kernel_size, second_kernel_size], "BHWCLKRSUV")]

    if define_tiling_space:
        ctx.define_split(b, nparts=2*levels)
        ctx.define_split(h, nparts=2*levels)
        ctx.define_split(w, nparts=2*levels)
        ctx.define_split(c, nparts=2*levels)
        ctx.define_split(l, nparts=2*levels)
        ctx.define_split(k, nparts=2*levels)
        ctx.define_split(r, nparts=2*levels)
        ctx.define_split(s, nparts=2*levels)
        ctx.define_split(u, nparts=2*levels)
        ctx.define_split(v, nparts=2*levels)

        factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_w = ctx.get_split(w)
        factors_c = ctx.get_split(c)
        factors_l = ctx.get_split(l)
        factors_k = ctx.get_split(k)
        factors_r = ctx.get_split(r)
        factors_s = ctx.get_split(s)
        factors_u = ctx.get_split(u)
        factors_v = ctx.get_split(v)
    else:
        factors_b = [dir.Var("int32") for i in range(2*levels)]
        factors_h = [dir.Var("int32") for i in range(2*levels)]
        factors_w = [dir.Var("int32") for i in range(2*levels)]
        factors_c = [dir.Var("int32") for i in range(2*levels)]
        factors_l = [dir.Var("int32") for i in range(2*levels)]
        factors_k = [dir.Var("int32") for i in range(2*levels)]
        factors_r = [dir.Var("int32") for i in range(2*levels)]
        factors_s = [dir.Var("int32") for i in range(2*levels)]
        factors_u = [dir.Var("int32") for i in range(2*levels)]
        factors_v = [dir.Var("int32") for i in range(2*levels)]

    sub_b = ctx.split(b, factors=factors_b + [1])
    sub_h = ctx.split(h, factors=factors_h + [1])
    sub_w = ctx.split(w, factors=factors_w + [1])
    sub_c = ctx.split(c, factors=factors_c + [1])
    sub_l = ctx.split(l, factors=factors_l + [1])
    sub_k = ctx.split(k, factors=factors_k + [1])
    sub_r = ctx.split(r, factors=factors_r + [1])
    sub_s = ctx.split(s, factors=factors_s + [1])
    sub_u = ctx.split(u, factors=factors_u + [1])
    sub_v = ctx.split(v, factors=factors_v + [1])

    loop_b = [b, *sub_b]
    loop_h = [h, *sub_h]
    loop_w = [w, *sub_w]
    loop_c = [c, *sub_c]
    loop_k = [k, *sub_k]
    loop_l = [l, *sub_l]
    loop_r = [r, *sub_r]
    loop_s = [s, *sub_s]
    loop_u = [u, *sub_u]
    loop_v = [v, *sub_v]

    tA = Conv2d(ctx, tI, tW1, loop_b, loop_h, loop_w,
                loop_c, loop_l, loop_r, loop_s, levels)
    tB = Conv2d(ctx, tA, tW2, loop_b, loop_h, loop_w,
                loop_l, loop_k, loop_u, loop_v, levels)

    ctx.define_fuse(tB, levels)
    fuse_choice = ctx.get_fuse()
    fuse_choice.apply(tB, ctx)

    return [tB], [b, h, w, c, l, k, r, s, u, v]


def conv_chain_nchw_dataflow(ctx, tI, tW1, tW2, batch, height, width, in_channel, out_channel_1, out_channel_2, levels, define_tiling_space=True, second_kernel_size=3):
    """
    tA: [batch, height, width, in_channel]
    tB: [out_channel_1, kh, kw, in_channel]
    tC: [out_channel_2, kh, kw, out_channel_1]
    """
    b, h, w, c, l, k, r, s, u, v = [dir.Loop(x, name=y) for (x, y) in zip(
        [batch, height, width, in_channel, out_channel_1, out_channel_2, 3, 3, second_kernel_size, second_kernel_size], "BHWCLKRSUV")]

    if define_tiling_space:
        ctx.define_split(b, nparts=2*levels)
        ctx.define_split(h, nparts=2*levels)
        ctx.define_split(w, nparts=2*levels)
        ctx.define_split(c, nparts=2*levels)
        ctx.define_split(l, nparts=2*levels)
        ctx.define_split(k, nparts=2*levels)
        ctx.define_split(r, nparts=2*levels)
        ctx.define_split(s, nparts=2*levels)
        ctx.define_split(u, nparts=2*levels)
        ctx.define_split(v, nparts=2*levels)

        factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_w = ctx.get_split(w)
        factors_c = ctx.get_split(c)
        factors_l = ctx.get_split(l)
        factors_k = ctx.get_split(k)
        factors_r = ctx.get_split(r)
        factors_s = ctx.get_split(s)
        factors_u = ctx.get_split(u)
        factors_v = ctx.get_split(v)
    else:
        factors_b = [dir.Var("int32") for i in range(2*levels)]
        factors_h = [dir.Var("int32") for i in range(2*levels)]
        factors_w = [dir.Var("int32") for i in range(2*levels)]
        factors_c = [dir.Var("int32") for i in range(2*levels)]
        factors_l = [dir.Var("int32") for i in range(2*levels)]
        factors_k = [dir.Var("int32") for i in range(2*levels)]
        factors_r = [dir.Var("int32") for i in range(2*levels)]
        factors_s = [dir.Var("int32") for i in range(2*levels)]
        factors_u = [dir.Var("int32") for i in range(2*levels)]
        factors_v = [dir.Var("int32") for i in range(2*levels)]

    sub_b = ctx.split(b, factors=factors_b + [1])
    sub_h = ctx.split(h, factors=factors_h + [1])
    sub_w = ctx.split(w, factors=factors_w + [1])
    sub_c = ctx.split(c, factors=factors_c + [1])
    sub_l = ctx.split(l, factors=factors_l + [1])
    sub_k = ctx.split(k, factors=factors_k + [1])
    sub_r = ctx.split(r, factors=factors_r + [1])
    sub_s = ctx.split(s, factors=factors_s + [1])
    sub_u = ctx.split(u, factors=factors_u + [1])
    sub_v = ctx.split(v, factors=factors_v + [1])

    loop_b = [b, *sub_b]
    loop_h = [h, *sub_h]
    loop_w = [w, *sub_w]
    loop_c = [c, *sub_c]
    loop_k = [k, *sub_k]
    loop_l = [l, *sub_l]
    loop_r = [r, *sub_r]
    loop_s = [s, *sub_s]
    loop_u = [u, *sub_u]
    loop_v = [v, *sub_v]

    tA = Conv2d_nchw(ctx, tI, tW1, loop_b, loop_h, loop_w,
                     loop_c, loop_l, loop_r, loop_s, levels)
    tB = Conv2d_nchw(ctx, tA, tW2, loop_b, loop_h, loop_w,
                     loop_l, loop_k, loop_u, loop_v, levels)

    ctx.define_fuse(tB, levels)
    fuse_choice = ctx.get_fuse()
    fuse_choice.apply(tB, ctx)

    return [tB], [b, h, w, c, l, k, r, s, u, v]


def get_tileflow_conv_chain_dataflow(levels, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=True, layout="nhwc", second_kernel_size=3):
    def static_func(ctx):
        # use NameScope to allow the same name for different plan
        with dir.NameScope(only_capital=True):
            if layout == "nhwc":
                tI = dir.Tensor([batch, height, width, in_channel],
                                name="I", dtype="int16", ctx=ctx)
                tW1 = dir.Tensor([out_channel_1, 3, 3, in_channel],
                                 name="X", dtype="int16", ctx=ctx)
                tW2 = dir.Tensor([out_channel_2, second_kernel_size, second_kernel_size, out_channel_1],
                                 name="Y", dtype="int16", ctx=ctx)

                [tB], loops = conv_chain_dataflow(
                    ctx, tI, tW1, tW2, batch, height, width, in_channel, out_channel_1, out_channel_2, levels, define_tiling_space=define_tiling_space, second_kernel_size=second_kernel_size)
            elif layout == "nchw":
                tI = dir.Tensor([batch, in_channel, height, width],
                                name="I", dtype="int16", ctx=ctx)
                tW1 = dir.Tensor([out_channel_1, in_channel, 3, 3],
                                 name="X", dtype="int16", ctx=ctx)
                tW2 = dir.Tensor([out_channel_2, out_channel_1, 1, 1],
                                 name="Y", dtype="int16", ctx=ctx)

                [tB], loops = conv_chain_nchw_dataflow(
                    ctx, tI, tW1, tW2, batch, height, width, in_channel, out_channel_1, out_channel_2, levels, define_tiling_space=define_tiling_space, second_kernel_size=second_kernel_size)
            else:
                raise RuntimeError(f"Unknown layout {layout}.")
            return [tI, tW1, tW2], [tB], loops

    return static_func
