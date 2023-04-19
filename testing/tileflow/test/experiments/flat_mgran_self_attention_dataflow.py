
from tileflow import (Context, generate_workload, arch_lower, arch_build, sequential_work,
                      ops, get_space, generate_candidate, concurrent_work, get_edge_small, get_cloud_small, get_edge_large, get_cloud_large)

import domino.accelerator as acc
import domino.program_ir as dir
import domino.dse as dse
import domino.runtime as rt
from tqdm import tqdm
import time


def self_attention_compute(ctx, tQ, tK, tV, batch, num_heads, seq_len, hidden, levels):
    """
    tQ: [num_heads, seq_len, model_k]
    tK: [num_heads, model_k, seq_len]
    tV: [num_heads, seq_len, model_k]
    """
    assert levels in [2, 3]

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

    if levels == 2:
        ctx.define_split(b, nparts=3)
        ctx.define_split(h, nparts=3)
        ctx.define_split(n, nparts=2)
        ctx.define_split(k, nparts=2)
        
        if seq_len < 1024:
            ctx.define_split(m, nparts=3)
            ctx.define_split(l, nparts=2)
            factors_m = ctx.get_split(m)
            factors_l = ctx.get_split(l)
        else:
            assert seq_len % 512 == 0
            small_m = dir.Loop(512, name="M")
            small_l = dir.Loop(512, name="L")
            ctx.define_split(small_m, nparts=3)
            ctx.define_split(small_l, nparts=2)
            factors_m = ctx.get_split(small_m)
            factors_l = ctx.get_split(small_l)
            factors_m = [factors_m[0] * seq_len // 512, factors_m[1], factors_m[2]]
            factors_l = [factors_l[0] * seq_len // 512, factors_l[1]]

        factors_b = ctx.get_split(b)
        factors_h = ctx.get_split(h)
        factors_n = ctx.get_split(n)
        factors_k = ctx.get_split(k)
        
        # sub_b = ctx.split(b, factors=[factors_b[0], 1, factors_b[1]])
        # sub_h = ctx.split(h, factors=[factors_h[0], 1, factors_h[1]])
        # sub_m = ctx.split(m, factors=[factors_m[0], 1, factors_m[1]])
        sub_b = ctx.split(b, factors=factors_b)
        sub_h = ctx.split(h, factors=factors_h)
        sub_m = ctx.split(m, factors=factors_m)
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
    else:
        ctx.define_split(b, nparts=3)
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

        sub_b = ctx.split(
            b, factors=[factors_b[0], 1, factors_b[1], 1, factors_b[2]])
        sub_h = ctx.split(
            h, factors=[factors_h[0], 1, factors_h[1], 1, factors_h[2]])
        sub_m = ctx.split(
            m, factors=[factors_m[0], 1, factors_m[1], 1, factors_m[2]])
        sub_n = ctx.split(n, factors=[*factors_n, 1])
        sub_k = ctx.split(k, factors=[*factors_k, 1])
        sub_l = ctx.split(l, factors=[*factors_l, 1])

        b4, b3, b2, b1, b0 = sub_b
        h4, h3, h2, h1, h0 = sub_h
        m4, m3, m2, m1, m0 = sub_m
        n2, n1, n0 = sub_n
        k2, k1, k0 = sub_k
        l2, l1, l0 = sub_l

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


def run(levels, hw_config, batch, num_heads, seq_len, hidden, trials):

    def static_self_attention(ctx, B, H, M, N):
        # use NameScope to allow the same name for different plan
        with dir.NameScope(only_capital=True):
            tQ = dir.Tensor([B, H, M, N//H], name="Q", dtype="int16", ctx=ctx)
            tK = dir.Tensor([B, H, N//H, M], name="K", dtype="int16", ctx=ctx)
            tV = dir.Tensor([B, H, M, N//H], name="V", dtype="int16", ctx=ctx)
            [tF], loops = self_attention_compute(
                ctx, tQ, tK, tV, *[B, H, M, N], levels=levels)
            return [tQ, tK, tV], [tF], loops

    sp_beg = time.time()
    space = get_space(static_self_attention, [
                      batch, num_heads, seq_len, hidden])
    sp_end = time.time()
    print(f"Use {sp_end - sp_beg} s to get space.")

    epoch = 10
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
                space, static_self_attention, [batch, num_heads, seq_len, hidden])
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
