from .context import Context, arch_lower, arch_build
import domino.program_ir as dir
import domino.runtime as rt
from pebble import ProcessPool, ProcessExpired
import os
from tqdm import tqdm
import time

__all__ = [
    "get_space",
    "generate_candidate",
    "generate_workload",
    "evaluate_results",
    "sequential_work",
    "concurrent_work",
    "tuning",
    "inference"
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


def generate_workload(inputs, outputs, loops, ctx, fusion=True):
    graph = dir.make_prod_consum_graph(outputs[0])
    workload = graph.generate_tileflow_workload(loops, fusion=fusion)
    return workload


def evaluate_results(inputs, outputs, loops, ctx, workload, hw_config, unlimited_resource=False, print_log=False, tileflow_path=None, tileflow_mapper_metric=None):
    kernel = arch_lower(ctx)
    kernel = arch_build(kernel, target="tileflow")

    if print_log:
        print(workload)
        print(hw_config)
        print(kernel, flush=True)
    if tileflow_path is None:
        if "TILEFLOW_BIN_PATH" in os.environ:
            tileflow_path = os.environ["TILEFLOW_BIN_PATH"]
        else:
            tileflow_path = os.path.join(
                os.environ["HOME"], "TileFlow/build/bin/tileflow")
    if not os.path.exists(tileflow_path) or not os.path.isfile(tileflow_path):
        raise RuntimeError(
            "Can't find tileflow, try to set TILEFLOW_BIN_PATH environment variable.")
    perf = rt.run_tileflow(workload, hw_config, kernel,
                           tileflow_path=tileflow_path, unlimited_resource=unlimited_resource, tileflow_mapper_metric=tileflow_mapper_metric)
    return perf


INPUT_BUFFER = None


def worker(idx):
    global INPUT_BUFFER
    (hw_configs, candidates, fusion, unlimited_resource,
     print_log, tileflow_path, tileflow_mapper_metric) = INPUT_BUFFER
    hw_config = hw_configs[idx]
    inputs, outputs, loops, ctx = candidates[idx]
    workload = generate_workload(inputs, outputs, loops, ctx, fusion=fusion)
    perf = evaluate_results(inputs, outputs, loops, ctx, workload,
                            hw_config, unlimited_resource=unlimited_resource, print_log=print_log, tileflow_path=tileflow_path, tileflow_mapper_metric=tileflow_mapper_metric)
    return ctx.config_key, perf


def sequential_work(hw_configs, candidates, fusion=True, unlimited_resource=False, print_log=False, tileflow_path=None, tileflow_mapper_metric=None):
    global INPUT_BUFFER
    INPUT_BUFFER = (hw_configs, candidates, fusion,
                    unlimited_resource, print_log, tileflow_path, tileflow_mapper_metric)
    assert len(hw_configs) == len(candidates)
    results = []
    for i in range(len(candidates)):
        results.append(worker(i))
    return results


def concurrent_work(hw_configs, candidates, fusion=True, unlimited_resource=False, print_log=False, tileflow_path=None, tileflow_mapper_metric=None):
    global INPUT_BUFFER
    assert len(hw_configs) == len(candidates)
    INPUT_BUFFER = (hw_configs, candidates, fusion,
                    unlimited_resource, print_log, tileflow_path, tileflow_mapper_metric)
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


def metric(perf, metric_type):
    if perf["status_ok"]:
        if metric_type == "1e9/latency":
            return 1e9/(perf["Cycle"]+1e-5)
        if metric_type == "1e9/energy":
            return 1e9/(perf["Energy"]+1e-5)
        if metric_type == "1e9/EDP":
            return 1e9 / (perf["Energy"] * perf["Cycle"]+1e-5)
        if metric_type == "Utilization_L0":
            util = 0
            for k, v in perf.items():
                if k == "MEM::L0":
                    util = max(util, v)
            return util
        if metric_type == "Utilization_L1":
            util = 0
            for k, v in perf.items():
                if k == "MEM::L1":
                    util = max(util, v)
            return util
        if metric_type == "Utilization_L2":
            util = 0
            for k, v in perf.items():
                if k == "MEM::L2":
                    util = max(util, v)
            return util
        if metric_type == "Utilization_L3":
            util = 0
            for k, v in perf.items():
                if k == "MEM::L3":
                    util = max(util, v)
            return util
        else:
            raise NotImplementedError(f"Unkonwn metric {metric_type}")
    else:
        return 0


def tuning(hw_config, func, params, trials, metric_type, sequential=False, resource_check=True, debug=False, tileflow_path=None):
    print("Get space...")
    space = get_space(func, params)

    steps = (trials+9)//10
    epoch = 10
    if trials < 10:
        epoch = 1
        steps = trials
    results = []
    temporal_best_config_key = None
    temporal_best_perf = None
    temporal_best_score = 0
    for ep in tqdm(range(epoch)):
        ep_beg = time.time()
        hw_configs = []
        candidates = []
        st_beg = time.time()
        for i in range(steps):
            hw_configs.append(hw_config)
            candidate = generate_candidate(
                space, func, params)
            candidates.append(candidate)
        st_end = time.time()
        print(f"Use {st_end - st_beg} s to generate candidates.")
        if sequential:
            batch_results = sequential_work(
                hw_configs, candidates, unlimited_resource=not resource_check, print_log=debug, tileflow_path=tileflow_path, tileflow_mapper_metric=metric_type)
        else:
            batch_results = concurrent_work(
                hw_configs, candidates, unlimited_resource=not resource_check, print_log=debug, tileflow_path=tileflow_path, tileflow_mapper_metric=metric_type)
        wk_end = time.time()
        print(f"Use {wk_end - st_end} s to evaluate.")
        print("feedback to space")
        for (key, value) in tqdm(batch_results):
            if value["status_ok"]:
                score = metric(value, metric_type)
                space[key] = score
                if score > temporal_best_score:
                    temporal_best_score = score
                    temporal_best_perf = value
                    temporal_best_config_key = key
        results.extend(batch_results)
        ep_end = time.time()
        print(f"One Epoch Use Time: {ep_end - ep_beg} s.")
        print(f"Temporal Best Score: {temporal_best_score}")
        print(f"Temporal Best Perf: {temporal_best_perf}")
        print(f"Temporal Best Config Key: {temporal_best_config_key}")

    best_perf = None
    best_score = 0
    best_config_key = None
    best_config = None

    for i in tqdm(range(epoch * steps)):
        perf = results[i][1]
        if perf["status_ok"]:
            if best_score < metric(perf, metric_type):
                best_score = metric(perf, metric_type)
                best_perf = perf
                best_config_key = results[i][0]
                best_config = space[best_config_key]

    print("Best Config Key:", best_config_key)
    print("Best Score:", best_score)
    print("Best Config:", best_config)
    print("Best Perf:", best_perf)
    return best_perf, best_config_key, best_config


def inference(hw_config, func, params, metric_type, resource_check=True, debug=False, tileflow_path=None):
    ctx = Context()
    inputs, outputs, loops = func(ctx, *params)
    workload = generate_workload(inputs, outputs, loops, ctx, fusion=True)
    perf = evaluate_results(inputs, outputs, loops, ctx, workload,
                            hw_config, unlimited_resource=not resource_check, print_log=debug, tileflow_path=tileflow_path, tileflow_mapper_metric=metric_type)

    return perf, None, None
