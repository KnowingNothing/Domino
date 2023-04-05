import domino.program_ir as dir

def gemm_exp_gemm(ctx, tA, tB, tC, tD, tE, tF, M, N, K, L):
    m, n = [dir.SLoop(x, name=y) for (x, y) in zip([M, N], "MN")]
    k, l = [dir.RLoop(x, name=y) for (x, y) in zip([K, L], "KL")]
    m1, m0s, m0t = ctx.split(m, nparts=3, factors=[M//2, 2, 1])
    n1, n0s, n0t = ctx.split(n, nparts=3, factors=[N//2, 2, 1])
    k1, k0s, k0t = ctx.split(k, nparts=3, factors=[K//2, 2, 1])
    l1, l0s, l0t = ctx.split(l, nparts=3, factors=[L//2, 2, 1])
    
    with ctx.tile("L1", [m1, l1, n1]):
        with ctx.pipeline():
            with ctx.tile("L0", [m0s, l0s]):
                with ctx.tile("L0", [k, m0t, l0t]):
                    tC[m, l] = tC[m, l] + tA[m, k] * tB[k, l]
            with ctx.tile("L0", [m0s, l0s]):
                with ctx.tile("L0", [m0t, l0t]):
                    tD[m, l] = dir.exp(tC[m, l])
            with ctx.tile("L0", [m0s, n0s, l0s]):
                with ctx.tile("L0", [m0t, n0t, l0t]):
                    tF[m, n] = tD[m, l] * tE[l, n]
                
    ctx.spatial(m0s)
    ctx.spatial(l0s)
    ctx.spatial(n0s)
                
if __name__ == "__main__":
    ctx = dir.IRBuilderContext()
    ctx.set_target_tileflow()
    M = 512
    N = 64
    K = 64
    L = 512
    tA = dir.Tensor([M, K], name="A", dtype="float16")
    tB = dir.Tensor([K, L], name="B", dtype="float16")
    tC = dir.Tensor([M, L], name="C", dtype="float16")
    tD = dir.Tensor([M, L], name="D", dtype="float16")
    tE = dir.Tensor([L, N], name="E", dtype="float16")
    tF = dir.Tensor([M, N], name="F", dtype="float16")
    
    def static_func(ctx, tA, tB, tC, tD, tE, tF):
        return gemm_exp_gemm(ctx, tA, tB, tC, tD, tE, tF, M, N, K, L)
    kernel = dir.arch_lower(static_func, [tA, tB, tC, tD, tE, tF], ctx=ctx)
    dir.print_ir(kernel)
    kernel = dir.arch_build(kernel, ctx=ctx, target="tileflow")
    print(kernel)