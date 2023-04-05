import domino.program_ir as dir


def self_attention(ctx, Q, K, V, Output, batch_size, num_heads, seq_len, model_k):
    b, h, m, n = [dir.SLoop(x, name=y) for (x, y) in zip(
        [batch_size, num_heads, seq_len, model_k], "bhmn")]
    r1, r2, r3, r4 = [dir.RLoop(x, name=y) for (x, y) in zip(
        [model_k, seq_len, seq_len, seq_len], ["r1", "r2", "r3", "r4"])]

    b2, b1, b0 = ctx.split(b, nparts=3, factors=[batch_size//4, 2, 2])
    h2, h1, h0 = ctx.split(h, nparts=3, factors=[num_heads//2, 1, 2])
    m2, m1, m0 = ctx.split(m, nparts=3, factors=[seq_len//4, 1, 4])
    n2, n1, n0 = ctx.split(n, nparts=3, factors=[model_k//4, 1, 4])

    r11, r10 = ctx.split(r1, nparts=2, factors=[model_k//16, 16])
    r21, r20 = ctx.split(r2, nparts=2, factors=[seq_len//16, 16])
    r31, r30 = ctx.split(r3, nparts=2, factors=[seq_len//16, 16])
    r41, r40 = ctx.split(r4, nparts=2, factors=[seq_len//16, 16])

    l = dir.SLoop(r41.extent * r40.extent, name="l")
    l1, l0 = ctx.split(l, nparts=2, factors=[l.extent//16, 16])

    L = ctx.Array([batch_size, num_heads, seq_len, seq_len],
                   name="L", dtype=Q.dtype)
    M = ctx.Array([batch_size, num_heads, seq_len], name="M", dtype=L.dtype)

    # ctx.spatial(m0)
    # ctx.spatial(n0)
    # ctx.spatial(b1)
    # ctx.spatial(h1)
    for loop in [m0, l0, r10, h0, r20, r30]:
        ctx.spatial(loop)

    def L1_bmm1():
        with ctx.tile("L1", [b1, h1, m1, l1]):
            # with ctx.tile("L0", [b0, h0, m0, l0]):
            #     L[b, h, m, l] = dir.const(0, L.dtype)
            with ctx.tile("L0", [r11, b0, h0, m0, l0, r10]):
                L[b, h, m, l] = L[b, h, m, l] + \
                    Q[b, h, m, r1] * K[b, h, l, r1]

    def L1_softmax():
        def L1_max():
            with ctx.tile("L1", [b1, h1, m1]):
                # with ctx.tile("L0", [b0, h0, m0]):
                #     M[b, h, m] = L[b, h, m, 0]
                with ctx.tile("L0", [r21, b0, h0, m0, r20]):
                    M[b, h, m] = dir.max(M[b, h, m], L[b, h, m, r2])

        def L1_sub_exp():
            with ctx.tile("L1", [b1, h1, m1, l1]):
                with ctx.tile("L0", [b0, h0, m0, l0]):
                    L[b, h, m, l] = L[b, h, m, l] - M[b, h, m]
                    L[b, h, m, l] = dir.exp(L[b, h, m, l])

        def L1_sum():
            with ctx.tile("L1", [b1, h1, m1]):
                # with ctx.tile("L0", [b0, h0, m0]):
                #     M[b, h, m] = dir.const(0, L.dtype)
                with ctx.tile("L0", [r31, b0, h0, m0, l0, r30]):
                    M[b, h, m] = M[b, h, m] + L[b, h, m, r3]

        def L1_div():
            with ctx.tile("L1", [b1, h1, m1, l1]):
                with ctx.tile("L0", [b0, h0, m0, l0]):
                    L[b, h, m, l] = L[b, h, m, l] / M[b, h, m]
        with ctx.sequential():
            L1_max()
            L1_sub_exp()
            L1_sum()
            L1_div()

    def L1_bmm2():
        with ctx.tile("L1", [b1, h1, m1, n1]):
            # with ctx.tile("L0", [b0, h0, m0, n0]):
            #     Output[b, h, m, n] = dir.const(0, Output.dtype)
            with ctx.tile("L0", [r41, b0, h0, m0, n0, r40]):
                Output[b, h, m, n] = Output[b, h, m, n] + \
                    L[b, h, m, r4] * V[b, h, n, r4]

    with ctx.tile("L2", [b2, h2, m2, n2]):
        with ctx.pipeline():
            L1_bmm1()
            L1_softmax()
            L1_bmm2()


if __name__ == "__main__":
    ctx = dir.IRBuilderContext()
    ctx.set_target_tileflow()
    batch_size = 64
    seq_len = 1024
    num_heads = 12
    model_k = 64
    Q = dir.Tensor([batch_size, num_heads, seq_len, model_k],
                   name="Q", dtype="float16")
    K = dir.Tensor([batch_size, num_heads, seq_len, model_k],
                   name="K", dtype="float16")
    V = dir.Tensor([batch_size, num_heads, model_k, seq_len],
                   name="V", dtype="float16")
    Output = dir.Tensor([batch_size, num_heads, seq_len,
                        model_k], name="Output", dtype="float16")
    # self_attention(ctx, Q, K, V, Output, batch_size,
    #                num_heads, seq_len, model_k)

    def static_func(ctx, Q, K, V, Output):
        return self_attention(ctx, Q, K, V, Output, batch_size, num_heads, seq_len, model_k)
    kernel = dir.arch_lower(static_func, [Q, K, V, Output], ctx=ctx)
    dir.print_ir(kernel)
    kernel = dir.arch_build(kernel, ctx=ctx, target="tileflow")
    print(kernel)