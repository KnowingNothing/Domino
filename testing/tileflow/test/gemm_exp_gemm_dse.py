import domino.program_ir as dir
import domino.analysis as ana
import domino.accelerator as acc
import domino.runtime as rt
from tileflow import Context, arch_lower, arch_build, register_workload
import domino.dse as dse
from pebble import ProcessPool, ProcessExpired
from tqdm import tqdm
import os


def get_hardware(levels):
    if levels == 2:
        MAC = acc.ALU(name="mac", alu_class="intmac",
                      datawidth=16, meshX=256, instance=256)
        Reg = acc.Buffer(name="L0", instance=16, buffer_class="regfile", width=16, depth=64,
                         meshX=16, word_bits=8, read_bandwidth=2, write_bandwidth=2, technology="45nm")
        PE = acc.Engine(name="PE")
        PE.add_local(Reg, MAC)
        Core = acc.Engine(name="System")
        L1 = acc.Buffer(name="L1", buffer_class="DRAM",
                        width=256, block_size=32, word_bits=8)
        Core.add_level(PE)
        Core.add_local(L1)
        Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
        Acc.set_hardware_level(Core)
        return Acc
    elif levels == 3:
        MAC = acc.ALU(name="mac", alu_class="intmac",
                      datawidth=16, meshX=256, instance=256)
        Reg = acc.Buffer(name="L0", instance=16, buffer_class="regfile", width=16, depth=64,
                         meshX=16, word_bits=8, read_bandwidth=2, write_bandwidth=2, technology="45nm")
        L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=32,
                        depth=128, meshX=4, word_bits=8, instance=4)
        PE = acc.Engine(name="PE")
        PE.add_local(L1, Reg, MAC)
        Core = acc.Engine(name="System")
        L2 = acc.Buffer(name="L2", buffer_class="DRAM",
                        width=256, block_size=32, word_bits=8)
        Core.add_level(PE)
        Core.add_local(L2)
        Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
        Acc.set_hardware_level(Core)
        return Acc
    else:
        raise NotImplementedError()


def GeneralOperator(ctx, tensors, loops, levels, fbody):
    for loop in loops:
        assert len(loop) == 2 * levels, f"{len(loop)} vs {2 * levels}"
    org_loops = [x[0] for x in loops]

    def level_operator(lx):
        idx = int((levels - lx) * 2 + 1)
        if lx <= 1:
            with ctx.tile(f"L{lx-1}", [x[idx] for x in loops], "Temporal"):
                fbody(*tensors, *org_loops)
        else:
            with ctx.tile(f"L{lx-1}", [x[idx] for x in loops], "Temporal"):
                with ctx.tile(f"L{lx-1}", [x[idx+1] for x in loops], "Spatial"):
                    level_operator(lx-1)

    level_operator(levels)


def Gemm(ctx, tA, tB, loop_m, loop_l, loop_k, levels):
    tC = dir.Tensor([M, L], name="C", dtype="int16", ctx=ctx)

    def gemm_func(A, B, C, m, l, k):
        C[m, l] = C[m, l] + A[m, k] * B[k, l]

    GeneralOperator(ctx, [tA, tB, tC], [
                    loop_m, loop_l, loop_k], levels, gemm_func)
    return tC


def exp(ctx, tA, loop_m, loop_l, levels):
    tB = dir.Tensor([M, L], name="B", dtype="int16", ctx=ctx)

    def exp_func(A, B, m, l):
        B[m, l] = dir.exp(A[m, l])

    GeneralOperator(ctx, [tA, tB], [loop_m, loop_l], levels, exp_func)
    return tB


def gemm_exp_gemm_compute(ctx, tA, tB, tE, M, N, K, L, levels):
    m, n = [dir.Loop(x, name=y) for (x, y) in zip([M, N], "MN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([K, L], "KL")]

    ctx.define_split(m, nparts=2*levels-1)
    ctx.define_split(n, nparts=2*levels-1)
    ctx.define_split(k, nparts=2*levels-1)
    ctx.define_split(l, nparts=2*levels-1)
    factors_m = ctx.get_split(m)
    factors_n = ctx.get_split(n)
    factors_k = ctx.get_split(k)
    factors_l = ctx.get_split(l)
    sub_m = ctx.split(m, factors=factors_m)
    sub_n = ctx.split(n, factors=factors_n)
    sub_k = ctx.split(k, factors=factors_k)
    sub_l = ctx.split(l, factors=factors_l)

    loop_m = [m, *sub_m]
    loop_n = [n, *sub_n]
    loop_k = [k, *sub_k]
    loop_l = [l, *sub_l]

    tC = Gemm(ctx, tA, tB, loop_m, loop_l, loop_k, levels=levels)
    tD = exp(ctx, tC, loop_m, loop_l, levels=levels)
    tD = exp(ctx, tD, loop_m, loop_l, levels=levels)
    tF = Gemm(ctx, tD, tE, loop_m, loop_n, loop_l, levels=levels)

    ctx.define_fuse(tF, levels)
    fuse_choice = ctx.get_fuse()
    fuse_choice.apply(tF, ctx)

    return [tF], [m, n, k, l]


# create space
def get_space(func, params, constraint=None):
    ctx = Context()
    ctx.enable_tuning()
    func(ctx, *params)
    ctx.space.set_constraint(constraint)
    return ctx.space


def generate_candidate(space, func, params):
    ctx = Context()
    ctx.set_space(space)
    ctx.enable_tuning()
    inputs, outputs, loops = func(ctx, *params)
    return (inputs, outputs, loops, ctx)


def generate_workload(inputs, outputs, loops, ctx):
    graph = dir.make_prod_consum_graph(outputs[0])
    workload = graph.generate_tileflow_workload(loops)
    return workload


def evaluate_results(inputs, outputs, loops, ctx, workload, hw_config):
    kernel = arch_lower(ctx)
    kernel = arch_build(kernel, target="tileflow")

    # print(kernel)
    perf = rt.run_tileflow(workload, hw_config, kernel,
                           tileflow_path="/home/zchno/TileFlow/build/bin/tileflow")
    return perf


INPUT_BUFFER = None


def worker(idx):
    global INPUT_BUFFER
    (hw_configs, candidates) = INPUT_BUFFER
    hw_config = hw_configs[idx]
    inputs, outputs, loops, ctx = candidates[idx]
    workload = generate_workload(inputs, outputs, loops, ctx)
    perf = evaluate_results(inputs, outputs, loops, ctx, workload, hw_config)
    return ctx.config_key, perf


def concurrent_work(hw_configs, candidates):
    global INPUT_BUFFER
    assert len(hw_configs) == len(candidates)
    INPUT_BUFFER = (hw_configs, candidates)
    results = []
    with ProcessPool(os.cpu_count()) as pool:
        future = pool.map(worker, range(len(candidates)), timeout=100)
        iterator = future.result()

        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except TimeoutError as error:
                print("Evaluate Timeout.", flush=True)
                result = ({}, {"status_ok": False})
            except Exception as error:
                print(error)
                result = ({}, {"status_ok": False})
            results.append(result)
    return results


if __name__ == "__main__":
    # =------ HW Config ------=#
    levels = 2
    hw = get_hardware(levels)
    hw_config = acc.tileflow_accelerator_generator(hw)
    print(hw_config)

    M = 512
    N = 64
    K = 64
    L = 512

    @register_workload
    def static_gemm_exp_gemm(ctx, M, N, K, L):
        # use NameScope to allow the same name for different plan
        with dir.NameScope(only_capital=True):
            tA = dir.Tensor([M, K], name="A", dtype="int16", ctx=ctx)
            tB = dir.Tensor([K, L], name="B", dtype="int16", ctx=ctx)
            tE = dir.Tensor([L, N], name="E", dtype="int16", ctx=ctx)
            [tF], loops = gemm_exp_gemm_compute(
                ctx, tA, tB, tE, *[M, N, K, L], levels=levels)
            return [tA, tB, tE], [tF], loops

    space = get_space(static_gemm_exp_gemm, [M, N, K, L])

    epoch = 10
    steps = 100
    results = []
    for ep in tqdm(range(epoch)):
        hw_configs = []
        candidates = []
        for i in range(steps):
            hw_configs.append(hw_config)
            candidate = generate_candidate(
                space, static_gemm_exp_gemm, [M, N, K, L])
            candidates.append(candidate)
        batch_results = concurrent_work(hw_configs, candidates)
        print("feedback to space")
        for (key, value) in tqdm(batch_results):
            if value["status_ok"]:
                space[key] = 1/value["Cycle"]
        results.extend(batch_results)

    best_perf = float("inf")
    best_config_key = None
    best_config = None
    different_perf = set()

    for i in tqdm(range(epoch * steps)):
        perf = results[i][1]
        if perf["status_ok"]:
            if best_perf > perf["Cycle"]:
                best_perf = perf["Cycle"]
                best_config_key = results[i][0]
                best_config = space[best_config_key]
            different_perf.add(perf["Cycle"])

    print("Best Config Key:", best_config_key)
    print("Best Config:", best_config)
    print("Best Cycle:", best_perf)
    print(f"{len(different_perf)} performance results")
