
from tileflow import (Context, generate_workload, arch_lower, arch_build, sequential_work,
                      ops, get_space, generate_candidate, concurrent_work, get_edge_small, get_cloud_small, get_edge_large, get_cloud_large)

import domino.accelerator as acc
import domino.program_ir as dir
import domino.dse as dse
import domino.runtime as rt
from tqdm import tqdm
import time


def bmm4d_compute(ctx, tQ, tK, batch, num_heads, seq_len, hidden, levels):
    """
    tQ: [batch, num_heads, seq_len, model_k]
    tK: [batch, num_heads, model_k, seq_len]
    """

    model_k = hidden // num_heads
    b, h, m = [dir.Loop(x, name=y)
                  for (x, y) in zip([batch, num_heads, seq_len], "BHM")]
    k, l = [dir.Loop(x, name=y) for (x, y) in zip([model_k, seq_len], "KL")]

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
    ctx.define_split(k, nparts=levels*2)
    t1 = time.time()
    factors_b = ctx.get_split(b)
    factors_h = ctx.get_split(h)
    if seq_len < 1024:
        factors_m = ctx.get_split(m)
        factors_l = ctx.get_split(l)
    else:
        factors_m = ctx.get_split(small_m)
        factors_l = ctx.get_split(small_l)
    factors_k = ctx.get_split(k)
    # print(factors_h)
    # print(factors_m)
    # print(factors_n)
    # print(factors_k)

    t2 = time.time()
    # factors_l = [*[1 for i in range(2*levels-2)], seq_len]

    def helper(factors):
        # ret = []
        # for i in range(levels-1):
        #     ret.append(factors[i])
        #     ret.append(1)
        # ret.append(factors[levels-1])
        # ret.append(factors[levels])
        # ret.append(1)
        return factors + [1]
        return ret
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
    sub_k = ctx.split(k, factors=helper(factors_k))

    loop_b = [b, *sub_b]
    loop_h = [h, *sub_h]
    loop_m = [m, *sub_m]
    loop_k = [k, *sub_k]
    loop_l = [l, *sub_l]

    tC = ops.BatchGemm4D(ctx, tQ, tK, loop_b, loop_h, loop_m,
                         loop_l, loop_k, levels=levels)

    return [tC], [b, h, m, k, l]


def run(levels, hw_config, batch, num_heads, seq_len, hidden, trials):

    def static_bmm4d(ctx, B, H, M, N):
        # use NameScope to allow the same name for different plan
        with dir.NameScope(only_capital=True):
            tQ = dir.Tensor([B, H, M, N//H], name="Q", dtype="int16", ctx=ctx)
            tK = dir.Tensor([B, H, N//H, M], name="K", dtype="int16", ctx=ctx)
            [tF], loops = bmm4d_compute(
                ctx, tQ, tK, *[B, H, M, N], levels=levels)
            return [tQ, tK], [tF], loops

    sp_beg = time.time()
    space = get_space(static_bmm4d, [
                      batch, num_heads, seq_len, hidden])
    sp_end = time.time()
    print(f"Use {sp_end - sp_beg} s to get space.")

    epoch = trials // 100
    steps = 100
    results = []
    temporal_best_config_key = None
    temporal_best_perf = float("inf")
    for ep in tqdm(range(epoch)):
        ep_beg = time.time()
        hw_configs = []
        candidates = []
        st_beg = time.time()
        for i in range(steps):
            hw_configs.append(hw_config)
            candidate = generate_candidate(
                space, static_bmm4d, [batch, num_heads, seq_len, hidden])
            candidates.append(candidate)
        st_end = time.time()
        print(f"Use {st_end - st_beg} s to generate candidates.")
        batch_results = concurrent_work(hw_configs, candidates)
        # batch_results = sequential_work(hw_configs, candidates)
        wk_end = time.time()
        print(f"Use {wk_end - st_end} s to evaluate.")
        print("feedback to space")
        for (key, value) in tqdm(batch_results):
            if value["status_ok"]:
                space[key] = 1/(value["Cycle"]+1e-5)
                if value["Cycle"] < temporal_best_perf:
                    temporal_best_perf = value["Cycle"]
                    temporal_best_config_key = key
        results.extend(batch_results)
        ep_end = time.time()
        print(f"One Epoch Use Time: {ep_end - ep_beg} s.")
        print(f"Temporal Best Cycle: {temporal_best_perf}")
        print(f"Temporal Best Config Key: {temporal_best_config_key}")

    best_perf = {"Cycle": float("inf"), "Energy": float("inf")}
    best_config_key = None
    best_config = None
    different_perf = set()

    for i in tqdm(range(epoch * steps)):
        perf = results[i][1]
        if perf["status_ok"]:
            if best_perf["Cycle"] > perf["Cycle"]:
                best_perf = perf
                best_config_key = results[i][0]
                best_config = space[best_config_key]
            different_perf.add(perf["Cycle"])

    print("Best Config Key:", best_config_key)
    print("Best Config:", best_config)
    print("Best Cycle:", best_perf["Cycle"])
    print(f"{len(different_perf)} performance results")

    # config_key = '{"value": null, "children": {"H": {"value": 0, "children": {}}, "M": {"value": 54, "children": {}}, "N": {"value": 23, "children": {}}, "A": {"value": 34, "children": {}}, "fuse": {"value": 19923, "children": {}}}}'
    # config_key = dse.MultiDimKey.from_json(config_key)
    # space.set_config(config_key)

    # config = "{'H': [1, 1, 16], 'M': [512, 1, 1], 'N': [8, 4, 2], 'A': [32, 1, 2], 'fuse': Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), -1, Sequential); Tensor(G, [Const(16, int32),Const(512, int32),Const(512, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), 0, Sequential); Tensor(F, [Const(16, int32),Const(512, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), 0, Sequential); Tensor(E, [Const(16, int32),Const(512, int32),Const(512, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), 0, Pipeline); Tensor(D, [Const(16, int32),Const(512, int32),Const(512, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), 0, Parallel); Tensor(B, [Const(16, int32),Const(512, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), 0, Pipeline); Tensor(C, [Const(16, int32),Const(512, int32),Const(512, int32)], int16): (Tensor(I, [Const(16, int32),Const(512, int32),Const(64, int32)], int16), 0, Parallel); }"
    # print(space[config_key])

    # ctx = Context()
    # ctx.set_space(space)
    # inputs, outputs, loops = static_self_attention(ctx, batch, num_heads, seq_len, hidden)
    # workload = generate_workload(inputs, outputs, loops, ctx)
    # print(workload)
    # kernel = arch_lower(ctx)
    # kernel = arch_build(kernel, target="tileflow")
    # print(kernel)

    # # print(kernel)
    # perf = rt.run_tileflow(workload, hw_config, kernel,
    #                        tileflow_path="/home/zchno/TileFlow/build/bin/tileflow")
    # print(perf)
    return best_perf, best_config_key, best_config


if __name__ == "__main__":
    # =------ HW Config ------=#
    # levels = 3
    # hw = get_cloud_small()
    # hw_config = acc.tileflow_accelerator_generator(hw)
    # print(hw_config)

    # bert-base inference config
    batch = 1
    num_heads = 12
    seq_len = 512
    hidden = 12*64
    trials = 1000

    results = []
    for i, (levels, hw) in enumerate(zip([2, 2, 3, 3], [get_edge_small(), get_edge_large(), get_cloud_small(), get_cloud_large()])):
        for seq_len in [512, int(4*1024), int(16*1024), int(64*1024)]:
            print(f"Current Task for seq_len={seq_len}, hw_id={i}")
            hw_config = acc.tileflow_accelerator_generator(hw)
            perf, key, config = run(
                levels, hw_config, batch, num_heads, seq_len, hidden, trials)
            results.append((perf, key, config, seq_len, i))
            print(
                f"seq_len={seq_len}, hw_id={i}, Cycle={perf['Cycle']}, Energy={perf['Energy']}, key={key}, config={config}")
    print("###############################################")
    for res in results:
        perf, key, config, seq_len, i = res
        print(
            f"seq_len={seq_len}, hw_id={i}, Cycle={perf['Cycle']}, Energy={perf['Energy']}, key={key}, config={config}")
