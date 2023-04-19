
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

    if levels == 2:
        factors_b = [batch, 1, 1, 1]
        factors_m = [seq_len // 16, 1, 16, 1]
        factors_h = [num_heads // 4, 4, 1, 1]
        factors_l = [seq_len // 16, 1, 16, 1]
        factors_k = [model_k // 32, 1, 32, 1]
    else:
        factors_b = [batch, 1, 1, 1, 1, 1]
        factors_m = [seq_len // 256, 1, 4, 1, 64, 1]
        factors_h = [num_heads // 4, 1, 4, 1, 1, 1]
        factors_l = [seq_len // 256, 1, 4, 1, 64, 1]
        factors_k = [model_k // 32, 1, 32, 1, 1, 1]

    t2 = time.time()
    # factors_l = [*[1 for i in range(2*levels-2)], seq_len]

    def helper(factors):
        return factors + [1]

    sub_b = ctx.split(b, factors=helper(factors_b))
    sub_h = ctx.split(h, factors=helper(factors_h))
    sub_m = ctx.split(m, factors=helper(factors_m))
    sub_l = ctx.split(l, factors=helper(factors_l))
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
        
    ctx = Context()

    inputs, outputs, loops = static_bmm4d(ctx, batch, num_heads, seq_len, hidden)

    graph = dir.make_prod_consum_graph(outputs[0])
    workload = graph.generate_tileflow_workload(loops)

    kernel = arch_lower(ctx)
    kernel = arch_build(kernel, target="tileflow")

    # print(workload)
    # print(hw_config)
    # print(kernel)
    perf = rt.run_tileflow(workload, hw_config, kernel,
                           tileflow_path="/home/zchno/TileFlow/build/bin/tileflow")
    
    return perf


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
            perf = run(
                levels, hw_config, batch, num_heads, seq_len, hidden, trials)
            results.append((perf, seq_len, i))
            print(
                f"seq_len={seq_len}, hw_id={i}, Cycle={perf['Cycle']}, Energy={perf['Energy']}")
    print("###############################################")
    for res in results:
        perf, seq_len, i = res
        print(
            f"seq_len={seq_len}, hw_id={i}, Cycle={perf['Cycle']}, Energy={perf['Energy']}")
