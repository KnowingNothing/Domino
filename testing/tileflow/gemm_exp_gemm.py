import domino.program_ir as dir
import domino.analysis as ana


def get_hardware():
    empty = dir.AtomBlock(dir.Evaluate(0))
    null = dir.Var("null")
    compute = dir.AttrBlock("meshY", null, 4, empty)
    compute = dir.AttrBlock("meshX", null, 4, compute)
    compute = dir.AttrBlock("datawidth", null, 16, compute)
    compute = dir.AttrBlock("class", null, "intmac", compute)
    compute = dir.AttrBlock("name", null, "mac[0..15]", compute)
    register = dir.AttrBlock("write_bandwidth", null, 2, empty)
    register = dir.AttrBlock("read_bandwidth", null, 2, register)
    register = dir.AttrBlock("word-bits", null, 8, register)
    register = dir.AttrBlock("width", null, 16, register)
    register = dir.AttrBlock("depth", null, 64, register)
    register = dir.AttrBlock("meshY", null, 4, register)
    register = dir.AttrBlock("meshX", null, 4, register)
    register = dir.AttrBlock("class", null, "regfile", register)
    register = dir.AttrBlock("name", null, "L0[0..15]", register)
    raise NotImplementedError()

def Gemm(ctx, tA, tB, tC, loop_m, loop_l, loop_k):
    m, m1, m2, m3 = loop_m
    l, l1, l2, l3 = loop_l
    k, k1, k2, k3 = loop_k
    with ctx.tile("L1", [m1, l1, k1]):
        with ctx.tile("L1", [m2, l2, k2], "Spatial"):
            with ctx.tile("L0", [m3, l3, k3]):
                tC[m, l] = tC[m, l] + tA[m, k] * tB[k, l]
                
def exp(ctx, tA, tB, loop_m, loop_l):
    m, m1, m2, m3 = loop_m
    l, l1, l2, l3 = loop_l
    with ctx.tile("L1", [m1, l1]):
        with ctx.tile("L1", [m2, l2], "Spatial"):
            with ctx.tile("L0", [m3, l3]):
                tB[m, l] = dir.exp(tA[m, l])

def gemm_exp_gemm_compute(ctx, tA, tB, tC, tD, tE, tF, M, N, K, L):
    m, n = [dir.Loop(x, name=y) for (x, y) in zip([M, N], "MN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([K, L], "KL")]
    m1, m2, m3 = ctx.split(m, nparts=3, factors=[M//2, 2, 1])
    n1, n2, n3 = ctx.split(n, nparts=3, factors=[N//2, 2, 1])
    k1, k2, k3 = ctx.split(k, nparts=3, factors=[K//2, 2, 1])
    l1, l2, l3 = ctx.split(l, nparts=3, factors=[L//2, 2, 1])
    
    loop_m = [m, m1, m2, m3]
    loop_n = [n, n1, n2, n3]
    loop_k = [k, k1, k2, k3]
    loop_l = [l, l1, l2, l3]
    
    Gemm(ctx, tA, tB, tC, loop_m, loop_l, loop_k)
    exp(ctx, tC, tD, loop_m, loop_l)
    Gemm(ctx, tD, tE, tF, loop_m, loop_n, loop_l)
                

def gemm_exp_gemm_dataflow(ctx, tA, tB, tC, tD, tE, tF, M, N, K, L):
    m, n = [dir.Loop(x, name=y) for (x, y) in zip([M, N], "MN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([K, L], "KL")]
    m1, m0s, m0t = ctx.split(m, nparts=3, factors=[M//2, 2, 1])
    n1, n0s, n0t = ctx.split(n, nparts=3, factors=[1, 2, N//2])
    k1, k0s, k0t = ctx.split(k, nparts=3, factors=[1, 1, K])
    l1, l0s, l0t = ctx.split(l, nparts=3, factors=[L//2, 2, 1])
    
    with ctx.tile("L1", [m1, l1, k1, n1]):
        with ctx.pipeline():
            with ctx.tile("L1", [m0s, l0s, k0s], "Spatial"):
                with ctx.tile("L0", [k0t, m0t, l0t]):
                    tC[m, l] = tC[m, l] + tA[m, k] * tB[k, l]
            with ctx.tile("L1", [m0s, l0s], "Spatial"):
                with ctx.tile("L0", [m0t, l0t]):
                    tD[m, l] = dir.exp(tC[m, l])
            with ctx.tile("L1", [m0s, n0s, l0s], "Spatial"):
                with ctx.tile("L0", [m0t, n0t, l0t]):
                    tF[m, n] = tD[m, l] * tE[l, n]
                
def test_gemm_exp_gemm_dataflow():
    ctx = dir.IRBuilderContext()
    ctx.set_target_tileflow()
    M = 512
    N = 64
    K = 64
    L = 512
    tA = dir.Tensor([M, K], name="A", dtype="int16")
    tB = dir.Tensor([K, L], name="B", dtype="int16")
    tC = dir.Tensor([M, L], name="C", dtype="int16")
    tD = dir.Tensor([M, L], name="D", dtype="int16")
    tE = dir.Tensor([L, N], name="E", dtype="int16")
    tF = dir.Tensor([M, N], name="F", dtype="int16")
    
    def static_func(ctx, tA, tB, tC, tD, tE, tF):
        return gemm_exp_gemm_dataflow(ctx, tA, tB, tC, tD, tE, tF, M, N, K, L)
    kernel = dir.arch_lower(static_func, [tA, tB, tC, tD, tE, tF], ctx=ctx)
    # dir.print_ir(kernel)
    kernel = dir.arch_build(kernel, ctx=ctx, target="tileflow")
    print(kernel)
    

if __name__ == "__main__":
    ctx = dir.IRBuilderContext()
    ctx.set_target_tileflow()
    M = 512
    N = 64
    K = 64
    L = 512
    tA = dir.Tensor([M, K], name="A", dtype="int16", ctx=ctx)
    tB = dir.Tensor([K, L], name="B", dtype="int16", ctx=ctx)
    tC = dir.Tensor([M, L], name="C", dtype="int16", ctx=ctx)
    tD = dir.Tensor([M, L], name="D", dtype="int16", ctx=ctx)
    tE = dir.Tensor([L, N], name="E", dtype="int16", ctx=ctx)
    tF = dir.Tensor([M, N], name="F", dtype="int16", ctx=ctx)
    # arrays = [dir.Array(ctx, t.var, t.shape) for t in [tA, tB, tC, tD, tE, tF]]
    arrays = [tA, tB, tC, tD, tE, tF]
    gemm_exp_gemm_compute(ctx, *arrays, *[M, N, K, L])
    # print(ctx.stack[0].children[0].children[0].ctx)
    # graph = dir.make_prod_consum_graph(tF)
    # print(graph.nodes)
    # print(graph.feed_links)
    # dom = graph.dominators()
    # print(dom)
    # ret = ctx.build()
    # dir.print_ir(ret)
    # ana.merge_tree(ctx.stack[0], ctx.stack[0].children[2], ctx.stack[0].children[1], ctx.stack[0].children[2].children[0])
    # # ctx.stack[0].children = [ctx.stack[0].children[0], ctx.stack[0].children[2]]
    # ret = ctx.build()
    # dir.print_ir(ret)
    # ana.merge_tree(ctx.stack[0], ctx.stack[0].children[1], ctx.stack[0].children[0], ctx.stack[0].children[1].children[0].children[0])
    # ret = ctx.build()
    # dir.print_ir(ret)
    plans = ana.generate_fusion_plans(tF, 2)
    
    for plan in plans:
        print(plan)
        ctx = dir.IRBuilderContext()
        ctx.set_target_tileflow()
        for t in arrays:
            t.ctx = ctx
        gemm_exp_gemm_compute(ctx, *arrays, *[M, N, K, L])
        plan.apply(tF, ctx)
        # dir.print_ir(ctx.build())