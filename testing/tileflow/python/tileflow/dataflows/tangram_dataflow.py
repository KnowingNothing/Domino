import domino.program_ir as dir


def conv3x3_conv3x3_dataflow_2levels(ctx, tI, tW1, tW2, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=True):
    """
    tA: [batch, height, width, in_channel]
    tB: [out_channel_1, kh, kw, in_channel]
    tC: [out_channel_2, kh, kw, out_channel_1]
    """
    b, h, w, c, l, k, r, s, u, v = [dir.Loop(x, name=y) for (x, y) in zip(
        [batch, height, width, in_channel, out_channel_1, out_channel_2, 3, 3, 3, 3], "BHWCLKRSUV")]

    tA = dir.Tensor([batch, height, width, out_channel_1],
                    name="A", dtype="int16", ctx=ctx)
    tB = dir.Tensor([batch, height, width, out_channel_2],
                    name="B", dtype="int16", ctx=ctx)

    if define_tiling_space:
        # ctx.define_split(b, nparts=3)
        ctx.define_split(h, nparts=2)
        ctx.define_split(w, nparts=2)
        # ctx.define_split(c, nparts=2)
        ctx.define_split(l, nparts=3)
        ctx.define_split(k, nparts=3)

        # factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_w = ctx.get_split(w)
        # factors_c = ctx.get_split(c)
        factors_l = ctx.get_split(l)
        factors_k = ctx.get_split(k)
    else:
        # factors_b = [dir.Var("int32") for i in range(3)]
        factors_h = [dir.Var("int32") for i in range(2)]
        factors_w = [dir.Var("int32") for i in range(2)]
        # factors_c = [dir.Var("int32") for i in range(3)]
        factors_l = [dir.Var("int32") for i in range(3)]
        factors_k = [dir.Var("int32") for i in range(3)]

    # sub_b = ctx.split(b, factors=factors_b)
    sub_h = ctx.split(h, factors=factors_h)
    sub_w = ctx.split(w, factors=factors_w)
    # sub_c = ctx.split(c, factors=factors_c)
    sub_l = ctx.split(l, factors=factors_l)
    sub_k = ctx.split(k, factors=factors_k)

    # b2, b1, b0 = sub_b
    h1, h0 = sub_h
    w1, w0 = sub_w
    # c2, c1, c0 = sub_c
    l2, l1, l0 = sub_l
    k2, k1, k0 = sub_k

    with ctx.tile("L2", [b, k2, l2], "Temporal"):
        with ctx.pipeline():
            with ctx.tile("L2", [k1, l1], "Spatial"):
                with ctx.tile("L1", [h1, w1, c, l0], "Temporal"):
                    with ctx.tile("L1", [h0, w0], "Spatial"):
                        with ctx.tile("L0", [r, s], "Temporal"):
                            tA[b, h, w, l] = tA[b, h, w, l] + \
                                tI[b, h + r, w + s, c] * tW1[l, r, s, c]
            with ctx.tile("L2", [k1, l1], "Spatial"):
                with ctx.tile("L1", [h1, w1, l0, k0], "Temporal"):
                    with ctx.tile("L1", [h0, w0], "Spatial"):
                        with ctx.tile("L0", [u, v], "Temporal"):
                            tB[b, h, w, k] = tB[b, h, w, k] + \
                                tA[b, h + u, w + v, l] * tW2[k, u, v, l]

    return [tB], [b, h, w, c, l, k, r, s, u, v]


def conv3x3_conv3x3_dataflow_3levels(ctx, tI, tW1, tW2, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=True):
    """
    tA: [batch, height, width, in_channel]
    tB: [out_channel_1, kh, kw, in_channel]
    tC: [out_channel_2, kh, kw, out_channel_1]
    """
    b, h, w, c, l, k, r, s, u, v = [dir.Loop(x, name=y) for (x, y) in zip(
        [batch, height, width, in_channel, out_channel_1, out_channel_2, 3, 3, 3, 3], "BHWCLKRSUV")]

    tA = dir.Tensor([batch, height, width, out_channel_1],
                    name="A", dtype="int16", ctx=ctx)
    tB = dir.Tensor([batch, height, width, out_channel_2],
                    name="B", dtype="int16", ctx=ctx)

    if define_tiling_space:
        # ctx.define_split(b, nparts=3)
        ctx.define_split(h, nparts=2)
        ctx.define_split(w, nparts=2)
        # ctx.define_split(c, nparts=2)
        ctx.define_split(l, nparts=5)
        ctx.define_split(k, nparts=5)

        # factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_w = ctx.get_split(w)
        # factors_c = ctx.get_split(c)
        factors_l = ctx.get_split(l)
        factors_k = ctx.get_split(k)
    else:
        # factors_b = [dir.Var("int32") for i in range(3)]
        factors_h = [dir.Var("int32") for i in range(2)]
        factors_w = [dir.Var("int32") for i in range(2)]
        # factors_c = [dir.Var("int32") for i in range(2)]
        factors_l = [dir.Var("int32") for i in range(5)]
        factors_k = [dir.Var("int32") for i in range(5)]

    # sub_b = ctx.split(b, factors=factors_b)
    sub_h = ctx.split(h, factors=factors_h)
    sub_w = ctx.split(w, factors=factors_w)
    # sub_c = ctx.split(c, factors=factors_c)
    sub_l = ctx.split(l, factors=factors_l)
    sub_k = ctx.split(k, factors=factors_k)

    # b2, b1, b0 = sub_b
    h1, h0 = sub_h
    w1, w0 = sub_w
    # c1, c0 = sub_c
    l4, l3, l2, l1, l0 = sub_l
    k4, k3, k2, k1, k0 = sub_k

    with ctx.tile("L3", [b, k4, l4], "Temporal"):
        with ctx.tile("L3", [k3, l3], "Spatial"):
            with ctx.tile("L2", [k2, l2], "Temporal"):
                with ctx.pipeline():
                    with ctx.tile("L2", [k1, l1], "Spatial"):
                        with ctx.tile("L1", [h1, w1, c, l0], "Temporal"):
                            with ctx.tile("L1", [h0, w0], "Spatial"):
                                with ctx.tile("L0", [r, s], "Temporal"):
                                    tA[b, h, w, l] = tA[b, h, w, l] + \
                                        tI[b, h + r, w + s, c] * tW1[l, r, s, c]
                    with ctx.tile("L2", [k1, l1], "Spatial"):
                        with ctx.tile("L1", [h1, w1, l0, k0], "Temporal"):
                            with ctx.tile("L1", [h0, w0], "Spatial"):
                                with ctx.tile("L0", [u, v], "Temporal"):
                                    tB[b, h, w, k] = tB[b, h, w, k] + \
                                        tA[b, h + u, w + v, l] * tW2[k, u, v, l]

    return [tB], [b, h, w, c, l, k, r, s, u, v]


def get_tangram_dataflow(levels, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=True):
    def static_self_attention(ctx):
        # use NameScope to allow the same name for different plan
        with dir.NameScope(only_capital=True):
            tI = dir.Tensor([batch, height, width, in_channel],
                            name="I", dtype="int16", ctx=ctx)
            tW1 = dir.Tensor([out_channel_1, 3, 3, in_channel],
                             name="X", dtype="int16", ctx=ctx)
            tW2 = dir.Tensor([out_channel_2, 3, 3, out_channel_1],
                             name="Y", dtype="int16", ctx=ctx)
            if levels == 2:
                [tB], loops = conv3x3_conv3x3_dataflow_2levels(
                    ctx, tI, tW1, tW2, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=define_tiling_space)
            elif levels == 3:
                [tB], loops = conv3x3_conv3x3_dataflow_3levels(
                    ctx, tI, tW1, tW2, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=define_tiling_space)
            else:
                raise NotImplementedError()
            return [tI, tW1, tW2], [tB], loops

    return static_self_attention
