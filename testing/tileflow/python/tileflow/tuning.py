from .context import Context, arch_lower, arch_build
import domino.program_ir as dir
import domino.runtime as rt
from pebble import ProcessPool, ProcessExpired
import os

__all__ = [
    "get_space",
    "generate_candidate",
    "generate_workload",
    "evaluate_results",
    "concurrent_work"
]

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
