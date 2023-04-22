import domino.program_ir as dir


def self_attention_dataflow_2levels(ctx, tQ, tK, tV, batch, num_heads, seq_len, hidden, define_tiling_space=True):
    """
    tQ: [num_heads, seq_len, model_k]
    tK: [num_heads, model_k, seq_len]
    tV: [num_heads, seq_len, model_k]
    """
    model_k = hidden // num_heads
    b, h, m, n = [dir.Loop(x, name=y) for (x, y) in zip(
        [batch, num_heads, seq_len, model_k], "BHMN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([model_k, seq_len], "KL")]

    tA = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="A", dtype="int16", ctx=ctx)
    tB = dir.Tensor([batch, num_heads, seq_len],
                    name="B", dtype="int16", ctx=ctx)
    tC = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="C", dtype="int16", ctx=ctx)
    tD = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="D", dtype="int16", ctx=ctx)
    tE = dir.Tensor([batch, num_heads, seq_len],
                    name="E", dtype="int16", ctx=ctx)
    tF = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="F", dtype="int16", ctx=ctx)
    tG = dir.Tensor([batch, num_heads, seq_len, model_k],
                    name="G", dtype="int16", ctx=ctx)

    if define_tiling_space:
        ctx.define_split(b, nparts=3)
        ctx.define_split(h, nparts=2)
        ctx.define_split(n, nparts=2)
        ctx.define_split(k, nparts=2)
        ctx.define_split(m, nparts=2)
        ctx.define_split(l, nparts=2)

        factors_m = ctx.get_split(m)
        factors_l = ctx.get_split(l)
        factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
    else:
        factors_b = [dir.Var("int32") for i in range(3)]
        factors_h = [dir.Var("int32") for i in range(3)]
        factors_m = [dir.Var("int32") for i in range(2)]
        factors_n = [dir.Var("int32") for i in range(2)]
        factors_k = [dir.Var("int32") for i in range(2)]
        factors_l = [dir.Var("int32") for i in range(2)]

    sub_b = ctx.split(b, factors=factors_b)
    sub_h = ctx.split(h, factors=[1, *factors_h])
    sub_m = ctx.split(m, factors=[1, *factors_m])
    sub_n = ctx.split(n, factors=[*factors_n, 1])
    sub_k = ctx.split(k, factors=[*factors_k, 1])
    sub_l = ctx.split(l, factors=[*factors_l, 1])

    b2, b1, b0 = sub_b
    h2, h1, h0 = sub_h
    m2, m1, m0 = sub_m
    n2, n1, n0 = sub_n
    k2, k1, k0 = sub_k
    l2, l1, l0 = sub_l

    with ctx.tile("L2", [b2, h2, m2], "Temporal"):
        with ctx.sequential():
            with ctx.pipeline():
                with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                    with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                        with ctx.tile("L1", [l1, k1], "Spatial"):
                            with ctx.tile("L0", [l0, k0], "Temporal"):
                                tA[b, h, m, l] = tA[b, h, m, l] + \
                                    tQ[b, h, m, k] * tK[b, h, k, l]
                with ctx.sequential():
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tB[b, h, m] = dir.max(
                                        tB[b, h, m], tA[b, h, m, l])
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tC[b, h, m, l] = tA[b, h, m, l] - \
                                        tB[b, h, m]
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tD[b, h, m, l] = dir.exp(
                                        tC[b, h, m, l])
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tE[b, h, m] = tE[b, h, m] + \
                                        tD[b, h, m, l]
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                            with ctx.tile("L1", [l1], "Spatial"):
                                with ctx.tile("L0", [l0], "Temporal"):
                                    tF[b, h, m, l] = tD[b, h, m, l] / \
                                        tE[b, h, m]
            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                with ctx.tile("L1", [b0, h0, m0, n2, l2], "Temporal"):
                    with ctx.tile("L1", [n1, l1], "Spatial"):
                        with ctx.tile("L0", [n0, l0], "Temporal"):
                            tG[b, h, m, n] = tG[b, h, m, n] + \
                                tF[b, h, m, l] * tV[b, h, l, n]

    return [tG], [b, h, m, n, k, l]


def self_attention_dataflow_3levels(ctx, tQ, tK, tV, batch, num_heads, seq_len, hidden, define_tiling_space=True):
    """
    tQ: [num_heads, seq_len, model_k]
    tK: [num_heads, model_k, seq_len]
    tV: [num_heads, seq_len, model_k]
    """

    model_k = hidden // num_heads
    b, h, m, n = [dir.Loop(x, name=y) for (x, y) in zip(
        [batch, num_heads, seq_len, model_k], "BHMN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([model_k, seq_len], "KL")]

    tA = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="A", dtype="int16", ctx=ctx)
    tB = dir.Tensor([batch, num_heads, seq_len],
                    name="B", dtype="int16", ctx=ctx)
    tC = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="C", dtype="int16", ctx=ctx)
    tD = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="D", dtype="int16", ctx=ctx)
    tE = dir.Tensor([batch, num_heads, seq_len],
                    name="E", dtype="int16", ctx=ctx)
    tF = dir.Tensor([batch, num_heads, seq_len, seq_len],
                    name="F", dtype="int16", ctx=ctx)
    tG = dir.Tensor([batch, num_heads, seq_len, model_k],
                    name="G", dtype="int16", ctx=ctx)

    if define_tiling_space:
        ctx.define_split(b, nparts=5)
        ctx.define_split(h, nparts=3)
        ctx.define_split(m, nparts=3)
        ctx.define_split(n, nparts=2)
        ctx.define_split(k, nparts=2)
        ctx.define_split(l, nparts=2)

        factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_m = ctx.get_split(m)
        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
        factors_l = ctx.get_split(l)
    else:
        factors_b = [dir.Var("int32") for i in range(5)]
        factors_h = [dir.Var("int32") for i in range(3)]
        factors_m = [dir.Var("int32") for i in range(3)]
        factors_n = [dir.Var("int32") for i in range(2)]
        factors_k = [dir.Var("int32") for i in range(2)]
        factors_l = [dir.Var("int32") for i in range(2)]

    b4, b3, b2, b1, b0 = ctx.split(b, factors=factors_b)
    h4, h3, h2, h1, h0 = ctx.split(h, factors=[1, 1, *factors_h])
    m4, m3, m2, m1, m0 = ctx.split(m, factors=[1, 1, *factors_m])
    n2, n1, n0 = ctx.split(n, factors=[*factors_n, 1])
    k2, k1, k0 = ctx.split(k, factors=[*factors_k, 1])
    l2, l1, l0 = ctx.split(l, factors=[*factors_l, 1])

    with ctx.tile("L3", [b4, h4, m4], "Temporal"):
        with ctx.tile("L3", [b3, h3, m3], "Spatial"):
            with ctx.tile("L2", [b2, h2, m2], "Temporal"):
                with ctx.sequential():
                    with ctx.pipeline():
                        with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                            with ctx.tile("L1", [b0, h0, m0, l2, k2], "Temporal"):
                                with ctx.tile("L1", [l1, k1], "Spatial"):
                                    with ctx.tile("L0", [l0, k0], "Temporal"):
                                        tA[b, h, m, l] = tA[b, h, m, l] + \
                                            tQ[b, h, m, k] * tK[b, h, k, l]
                        with ctx.sequential():
                            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                    with ctx.tile("L1", [l1], "Spatial"):
                                        with ctx.tile("L0", [l0], "Temporal"):
                                            tB[b, h, m] = dir.max(
                                                tB[b, h, m], tA[b, h, m, l])
                            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                    with ctx.tile("L1", [l1], "Spatial"):
                                        with ctx.tile("L0", [l0], "Temporal"):
                                            tC[b, h, m, l] = tA[b, h,
                                                                m, l] - tB[b, h, m]
                            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                    with ctx.tile("L1", [l1], "Spatial"):
                                        with ctx.tile("L0", [l0], "Temporal"):
                                            tD[b, h, m, l] = dir.exp(
                                                tC[b, h, m, l])
                            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                    with ctx.tile("L1", [l1], "Spatial"):
                                        with ctx.tile("L0", [l0], "Temporal"):
                                            tE[b, h, m] = tE[b, h, m] + \
                                                tD[b, h, m, l]
                            with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                                with ctx.tile("L1", [b0, h0, m0, l2], "Temporal"):
                                    with ctx.tile("L1", [l1], "Spatial"):
                                        with ctx.tile("L0", [l0], "Temporal"):
                                            tF[b, h, m, l] = tD[b, h,
                                                                m, l] / tE[b, h, m]
                    with ctx.tile("L2", [b1, h1, m1], "Spatial"):
                        with ctx.tile("L1", [b0, h0, m0, n2, l2], "Temporal"):
                            with ctx.tile("L1", [n1, l1], "Spatial"):
                                with ctx.tile("L0", [n0, l0], "Temporal"):
                                    tG[b, h, m, n] = tG[b, h, m, n] + \
                                        tF[b, h, m, l] * tV[b, h, l, n]

    return [tG], [b, h, m, n, k, l]


def get_flat_bgran_dataflow(levels, batch, num_heads, seq_len, hidden, define_tiling_space=True):
    def static_self_attention(ctx):
        # use NameScope to allow the same name for different plan
        B = batch
        H = num_heads
        M = seq_len
        N = hidden
        with dir.NameScope(only_capital=True):
            tQ = dir.Tensor([B, H, M, N//H], name="Q", dtype="int16", ctx=ctx)
            tK = dir.Tensor([B, H, N//H, M], name="K", dtype="int16", ctx=ctx)
            tV = dir.Tensor([B, H, M, N//H], name="V", dtype="int16", ctx=ctx)
            if levels == 2:
                [tF], loops = self_attention_dataflow_2levels(
                    ctx, tQ, tK, tV, *[B, H, M, N], define_tiling_space=define_tiling_space)
            elif levels == 3:
                [tF], loops = self_attention_dataflow_3levels(
                    ctx, tQ, tK, tV, *[B, H, M, N], define_tiling_space=define_tiling_space)
            else:
                raise NotImplementedError()
            return [tQ, tK, tV], [tF], loops

    return static_self_attention
