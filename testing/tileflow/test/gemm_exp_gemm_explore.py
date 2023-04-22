import domino.program_ir as dir
import domino.analysis as ana
import domino.accelerator as acc
import domino.runtime as rt
from tileflow import Context, arch_lower, arch_build


def get_hardware():
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=16, instance=16)
    Reg = acc.Buffer(name="L0", instance=16, buffer_class="regfile", width=16, depth=64,
                     meshX=16, word_bits=8, read_bandwidth=2, write_bandwidth=2, technology="45nm")
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    Core = acc.Engine(name="System")
    L1 = acc.Buffer(name="L1", buffer_class="DRAM",
                    width=256, depth=100000000, block_size=32, word_bits=8)
    Core.add_local(L1)
    Core.add_level(PE)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


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
    n1, n2, n3 = ctx.split(n, nparts=3, factors=[1, 1, N])
    k1, k2, k3 = ctx.split(k, nparts=3, factors=[K//2, 2, 1])
    l1, l2, l3 = ctx.split(l, nparts=3, factors=[L//2, 2, 1])

    loop_m = [m, m1, m2, m3]
    loop_n = [n, n1, n2, n3]
    loop_k = [k, k1, k2, k3]
    loop_l = [l, l1, l2, l3]

    Gemm(ctx, tA, tB, tC, loop_m, loop_l, loop_k)
    exp(ctx, tC, tD, loop_m, loop_l)
    Gemm(ctx, tD, tE, tF, loop_m, loop_n, loop_l)

    return [m, n, k, l]


if __name__ == "__main__":
    # =------ HW Config ------=#
    hw = get_hardware()
    hw_config = acc.tileflow_accelerator_generator(hw)

    ctx = Context()
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
    loops = gemm_exp_gemm_compute(ctx, *arrays, *[M, N, K, L])

    # =------ Workload Config ------=#
    graph = dir.make_prod_consum_graph(tF)
    workload = graph.generate_tileflow_workload(loops)

    # =------ Mapping Plans ------=#
    plans = ana.generate_fusion_plans(tF, 2)

    def static_func(ctx, tA, tB, tC, tD, tE, tF):
        # use NameScope to allow the same name for different plan
        with dir.NameScope():
            gemm_exp_gemm_compute(ctx, tA, tB, tC, tD, tE, tF, *[M, N, K, L])

    i = 0
    print(len(plans))
    for plan in plans[i:i+1]:
        plan.apply(tF, ctx)
        kernel = arch_lower(ctx)
        kernel = arch_build(kernel, target="tileflow")
        print(hw_config)
        print(workload)
        print(kernel)
        # print(kernel)
        results = rt.run_tileflow(workload, hw_config, kernel, tileflow_path="/home/zchno/TileFlow/build/bin/tileflow", unlimited_resource=False)
        print(results)
