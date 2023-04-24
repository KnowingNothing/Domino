
from tileflow import (get_edge_small, get_cloud_small,
                      get_edge_large, get_cloud_large, tuning)
import tileflow.dataflows as td

import domino.accelerator as acc
import argparse


def run(levels, hw_config, batch, height, width, in_channel, out_channel_1, out_channel_2, trials, metric_type, debug=False, resource_check=True, define_tiling_space=False):
    dataflow = td.get_tileflow_conv_chain_dataflow(
        levels, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=define_tiling_space)

    best_perf, best_config_key, best_config = tuning(
        hw_config, dataflow, [], trials, metric_type, sequential=debug, debug=debug, resource_check=resource_check)
    return best_perf, best_config_key, best_config


shapes = [
    # (in_channel, height, width, out_channel_1, out_channel_2)
    (64, 112, 112, 192, 128),  # Yolo
    (32, 147, 147, 64, 80),  # Inception-V3
    (64, 56, 56, 128, 64),  # Darknet-19
    (128, 28, 28, 256, 128),  # Darknet-19
    (16, 227, 227, 64, 16),  # Squeezenet-V1.1
]


"""example
python tileflow_conv_chainpy --metric=1e9/latency --trials 1 |& tee trace.log
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
                        help="Tuning trials [1000]", default=100)
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
        in_channel, height, width, out_channel_1, out_channel_2 = shape
        for i, (levels, hw) in enumerate(zip([2, 3], [get_edge_small(), get_cloud_small()])):
            print(
                f"Current Task for metric={args.metric}, batch={args.batch}, height={height}, width={width}, in_channel={in_channel}, out_channel_1={out_channel_1}, out_channel_2={out_channel_2}, hw_id={i}")
            hw_config = acc.tileflow_accelerator_generator(hw)
            perf, key, config = run(
                levels, hw_config, batch, height, width, in_channel, out_channel_1, out_channel_2, trials, metric_type=metric_type, debug=args.debug, resource_check=args.check_resource, define_tiling_space=args.define_tiling_space)
            results.append((perf, key, config, i))
            print(
                "batch,height,width,in_channel,out_channel_1,out_channel_2,metric,hw_id,key,config,perf")
            print(
                f"{batch},{height},{width},{in_channel},{out_channel_1},{out_channel_2},{metric_type},{i},{key},{config},{perf}")
        results_for_shape.append((shape, results))
    print("###############################################")
    for shape, results in results_for_shape:
        print(
            "batch,height,width,in_channel,out_channel_1,out_channel_2,metric,seq_len,hw_id,key,config,perf")
        for res in results:
            perf, key, config, i = res
            in_channel, height, width, out_channel_1, out_channel_2 = shape
            print(
                f"{batch},{height},{width},{in_channel},{out_channel_1},{out_channel_2},{metric_type},{i},{key},{config},{perf}")
