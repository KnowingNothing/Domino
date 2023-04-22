
from tileflow import (get_edge_small, get_cloud_small,
                      get_edge_large, get_cloud_large, tuning)
import tileflow.dataflows as td

import domino.accelerator as acc
import argparse


def run(levels, hw_config, batch, num_heads, seq_len, hidden, trials, metric_type, debug=False, resource_check=True, define_tiling_space=False):
    dataflow = td.get_tileflow_self_attention_dataflow(
        levels, batch, num_heads, seq_len, hidden, define_tiling_space=define_tiling_space)

    best_perf, best_config_key, best_config = tuning(
        hw_config, dataflow, [], trials, metric_type, sequential=debug, debug=debug, resource_check=resource_check)
    return best_perf, best_config_key, best_config


shapes = [
    # (num_heads, seq_len, hidden)
    (8, 512, 512),  # Bert-Small
    (12, 512, 768),  # Bert-Base
    (16, 512, 1024),  # Bert-Large
    (12, 256, 768),  # ViT-Base/14
    (16, 256, 1024),  # ViT-Large/14
    (16, 256, 1280),  # ViT-Huge/14
    (12, 196, 768),  # ViT-Base/16
    (16, 196, 1024),  # ViT-Large/16
    (16, 196, 1280),  # ViT-Huge/16
]


"""example
python tileflow_dataflow.py --metric=1e9/latency --trials 1 |& tee trace.log
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric", type=str, help="Evaluation metric type [1e9/latency, 1e9/energy, Utilization_L0, Utilization_L1, Utilization_L2, Utilization_L3]", default="1e9/latency")
    parser.add_argument("--batch", type=int,
                        help="Self attention batch size.", default=1)
    parser.add_argument("--begin", type=int,
                        help="Shape begin index [0]", default=0)
    parser.add_argument("--number", type=int,
                        help="Number of shapes evaluated [1]", default=1)
    parser.add_argument("--trials", type=int,
                        help="Tuning trials [1000]", default=1000)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--check_resource", default=False, action="store_true")
    parser.add_argument("--define_tiling_space",
                        default=False, action="store_true")

    args = parser.parse_args()
    batch = args.batch
    trials = args.trials

    metric_type = args.metric

    results_for_shape = []
    for shape in shapes[args.begin:args.begin+args.number]:
        results = []
        num_heads = shape[0]
        seq_len = shape[1]
        hidden = shape[2]
        for i, (levels, hw) in enumerate(zip([2, 2, 3, 3], [get_edge_small(), get_edge_large(), get_cloud_small(), get_cloud_large()])):
            print(
                f"Current Task for metric={args.metric}, batch={args.batch}, num_heads={num_heads}, seq_len={seq_len}, hidden={hidden} hw_id={i}")
            hw_config = acc.tileflow_accelerator_generator(hw)
            perf, key, config = run(
                levels, hw_config, batch, num_heads, seq_len, hidden, trials, metric_type=metric_type, debug=args.debug, resource_check=args.check_resource, define_tiling_space=args.define_tiling_space)
            results.append((perf, key, config, seq_len, i))
            print(
                "batch,seq_len,num_heads,hidden,metric,seq_len,hw_id,key,config,perf")
            print(
                f"{batch},{shape[1]},{shape[0]},{shape[2]},{metric_type},{seq_len},{i},{key},{config},{perf}")
        results_for_shape.append((shape, results))
    print("###############################################")
    for shape, results in results_for_shape:
        print(
            "batch,seq_len,num_heads,hidden,metric,seq_len,hw_id,key,config,perf")
        for res in results:
            perf, key, config, seq_len, i = res
            print(
                f"{batch},{shape[1]},{shape[0]},{shape[2]},{metric_type},{seq_len},{i},{key},{config},{perf}")
