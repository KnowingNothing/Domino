import domino.program_ir as dir
from ..ops import BatchGemm4D, softmax4D


def self_attention_dataflow(ctx, tQ, tK, tV, batch, num_heads, seq_len, hidden, levels, define_tiling_space=True):
    """
    tQ: [batch, num_heads, seq_len, model_k]
    tK: [batch, num_heads, model_k, seq_len]
    tV: [batch, num_heads, seq_len, model_k]
    """

    model_k = hidden // num_heads
    b, h, m, n = [dir.Loop(x, name=y)
                  for (x, y) in zip([batch, num_heads, seq_len, model_k], "BHMN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([model_k, seq_len], "KL")]

    if define_tiling_space:
        ctx.define_split(b, nparts=levels*2)
        if seq_len < 1024:
            ctx.define_split(m, nparts=levels*2)
            ctx.define_split(l, nparts=levels*2)
        else:
            assert seq_len % 512 == 0
            small_m = dir.Loop(512, name="M")
            ctx.define_split(small_m, nparts=levels*2)
            small_l = dir.Loop(512, name="L")
            ctx.define_split(small_l, nparts=levels*2)
        ctx.define_split(h, nparts=levels*2)
        ctx.define_split(n, nparts=levels*2)
        ctx.define_split(k, nparts=levels*2)

        factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        if seq_len < 1024:
            factors_m = ctx.get_split(m)
            factors_l = ctx.get_split(l)
        else:
            factors_m = ctx.get_split(small_m)
            factors_l = ctx.get_split(small_l)
        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
    else:
        factors_b = [dir.Var("int32") for i in range(levels*2)]
        factors_h = [dir.Var("int32") for i in range(levels*2)]
        factors_m = [dir.Var("int32") for i in range(levels*2)]
        factors_n = [dir.Var("int32") for i in range(levels*2)]
        factors_l = [dir.Var("int32") for i in range(levels*2)]
        factors_k = [dir.Var("int32") for i in range(levels*2)]

    def helper(factors):
        return factors + [1]

    sub_b = ctx.split(b, factors=helper(factors_b))
    sub_h = ctx.split(h, factors=helper(factors_h))
    if seq_len < 1024:
        sub_m = ctx.split(m, factors=helper(factors_m))
        sub_l = ctx.split(l, factors=helper(factors_l))
    else:
        factors_m = helper(factors_m)
        sub_m = ctx.split(
            m, factors=[factors_m[0] * seq_len // 512, *factors_m[1:]])
        factors_l = helper(factors_l)
        sub_l = ctx.split(
            l, factors=[factors_l[0] * seq_len // 512, *factors_l[1:]])
    sub_n = ctx.split(n, factors=helper(factors_n))
    sub_k = ctx.split(k, factors=helper(factors_k))

    loop_b = [b, *sub_b]
    loop_h = [h, *sub_h]
    loop_m = [m, *sub_m]
    loop_n = [n, *sub_n]
    loop_k = [k, *sub_k]
    loop_l = [l, *sub_l]

    tC = BatchGemm4D(ctx, tQ, tK, loop_b, loop_h, loop_m,
                     loop_l, loop_k, levels=levels)
    tD = softmax4D(ctx, tC, loop_b, loop_h, loop_m, loop_l, levels=levels)
    tF = BatchGemm4D(ctx, tD, tV, loop_b, loop_h, loop_m,
                     loop_n, loop_l, levels=levels)

    ctx.define_fuse(tF, levels)
    fuse_choice = ctx.get_fuse()
    fuse_choice.apply(tF, ctx)

    return [tF], [b, h, m, n, k, l]


def get_tileflow_self_attention_dataflow(levels, batch, num_heads, seq_len, hidden, define_tiling_space=True):
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
            [tF], loops = self_attention_dataflow(
                ctx, tQ, tK, tV, B, H, M, N, levels, define_tiling_space=define_tiling_space)
            return [tQ, tK, tV], [tF], loops

    return static_self_attention
