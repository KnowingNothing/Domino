
from tileflow import (Context, generate_workload, arch_lower, arch_build,
    ops, get_space, generate_candidate, concurrent_work, get_edge_small, get_cloud_small)

import domino.accelerator as acc
import domino.program_ir as dir
import domino.dse as dse
import domino.runtime as rt
from tqdm import tqdm


def self_attention_compute(ctx, tQ, tK, tV, num_heads, seq_len, hidden, levels):
    """
    tQ: [num_heads, seq_len, model_k]
    tK: [num_heads, model_k, seq_len]
    tV: [num_heads, seq_len, model_k]
    """

    model_k = hidden // num_heads
    h, m, n = [dir.Loop(x, name=y) for (x, y) in zip([num_heads, seq_len, model_k], "HMN")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([model_k, seq_len], "KL")]

    ctx.define_split(h, nparts=2*levels-1)
    ctx.define_split(m, nparts=2*levels-1)
    ctx.define_split(n, nparts=2*levels-1)
    ctx.define_split(k, nparts=2*levels-1)
    ctx.define_split(l, nparts=2*levels-1)
    factors_h = ctx.get_split(h)
    factors_m = ctx.get_split(m)
    factors_n = ctx.get_split(n)
    factors_k = ctx.get_split(k)
    # print(factors_h)
    # print(factors_m)
    # print(factors_n)
    # print(factors_k)
    factors_l = ctx.get_split(l)
    # factors_l = [*[1 for i in range(2*levels-2)], seq_len]
    sub_h = ctx.split(h, factors=factors_h)
    sub_m = ctx.split(m, factors=factors_m)
    sub_n = ctx.split(n, factors=factors_n)
    sub_k = ctx.split(k, factors=factors_k)
    sub_l = ctx.split(l, factors=factors_l)

    loop_h = [h, *sub_h]
    loop_m = [m, *sub_m]
    loop_n = [n, *sub_n]
    loop_k = [k, *sub_k]
    loop_l = [l, *sub_l]

    tC = ops.BatchGemm(ctx, tQ, tK, loop_h, loop_m, loop_l, loop_k, levels=levels)
    tD = ops.softmax3D(ctx, tC, loop_h, loop_m, loop_l, levels=levels)
    tF = ops.BatchGemm(ctx, tD, tV, loop_h, loop_m, loop_n, loop_l, levels=levels)

    ctx.define_fuse(tF, levels)
    fuse_choice = ctx.get_fuse()
    fuse_choice.apply(tF, ctx)

    return [tF], [h, m, n, k, l]


if __name__ == "__main__":
    # =------ HW Config ------=#
    levels = 3
    hw = get_cloud_small()
    hw_config = acc.tileflow_accelerator_generator(hw)
    print(hw_config)

    num_heads = 16
    seq_len = 512
    hidden = 1024

    def static_self_attention(ctx, H, M, N):
        # use NameScope to allow the same name for different plan
        with dir.NameScope(only_capital=True):
            tQ = dir.Tensor([H, M, N//H], name="Q", dtype="int16", ctx=ctx)
            tK = dir.Tensor([H, N//H, M], name="K", dtype="int16", ctx=ctx)
            tV = dir.Tensor([H, M, N//H], name="V", dtype="int16", ctx=ctx)
            [tF], loops = self_attention_compute(
                ctx, tQ, tK, tV, *[H, M, N], levels=levels)
            return [tQ, tK, tV], [tF], loops

    space = get_space(static_self_attention, [num_heads, seq_len, hidden])

    epoch = 10
    steps = 100
    results = []
    for ep in tqdm(range(epoch)):
        hw_configs = []
        candidates = []
        for i in range(steps):
            hw_configs.append(hw_config)
            candidate = generate_candidate(
                space, static_self_attention, [num_heads, seq_len, hidden])
            candidates.append(candidate)
        batch_results = concurrent_work(hw_configs, candidates)
        print("feedback to space")
        for (key, value) in tqdm(batch_results):
            if value["status_ok"]:
                space[key] = 1/(value["Cycle"]+1e-5)
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

    # config_key = '{"value": null, "children": {"H": {"value": 0, "children": {}}, "M": {"value": 54, "children": {}}, "N": {"value": 23, "children": {}}, "A": {"value": 34, "children": {}}, "fuse": {"value": 19923, "children": {}}}}'
    # config_key = dse.MultiDimKey.from_json(config_key)
    # space.set_config(config_key)

    # config = "{'H': [1, 1, 16], 'M': [512, 1, 1], 'N': [8, 4, 2], 'A': [32, 1, 2], 'fuse': Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), -1, Sequential); Tensor(G, [Const(16, int32),Const(512, int32),Const(512, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), 0, Sequential); Tensor(F, [Const(16, int32),Const(512, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), 0, Sequential); Tensor(E, [Const(16, int32),Const(512, int32),Const(512, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), 0, Pipeline); Tensor(D, [Const(16, int32),Const(512, int32),Const(512, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), 0, Parallel); Tensor(B, [Const(16, int32),Const(512, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), 0, Pipeline); Tensor(C, [Const(16, int32),Const(512, int32),Const(512, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), 0, Parallel); }"
    # print(space[config_key])

    # ctx = Context()
    # ctx.set_space(space)
    # inputs, outputs, loops = static_self_attention(ctx, num_heads, seq_len, hidden)
    # workload = generate_workload(inputs, outputs, loops, ctx)
    # print(workload)
    # kernel = arch_lower(ctx)
    # kernel = arch_build(kernel, target="tileflow")
    # print(kernel)

    # # print(kernel)
    # perf = rt.run_tileflow(workload, hw_config, kernel,
    #                        tileflow_path="/home/zchno/TileFlow/build/bin/tileflow")
    # print(perf)
