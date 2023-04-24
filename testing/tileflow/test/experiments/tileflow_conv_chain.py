
from tileflow import (get_edge_small, get_cloud_small,
                      get_edge_large, get_cloud_large, tuning, inference)
import tileflow.dataflows as td

import domino.accelerator as acc
import argparse
import json


def run(levels, hw_config, batch, height, width, in_channel, out_channel_1, out_channel_2, trials, metric_type, debug=False, resource_check=True, define_tiling_space=False, layout="nhwc"):
    dataflow = td.get_tileflow_conv_chain_dataflow(
        levels, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=define_tiling_space, layout=layout)

    best_perf, best_config_key, best_config = tuning(
        hw_config, dataflow, [], trials, metric_type, sequential=debug, debug=debug, resource_check=resource_check)
    return best_perf, best_config_key, best_config


def replay(logfile, dataflow_name, hw_id, batch, num_heads, seq_len, hidden, metric_type, debug=False, resource_check=True, define_tiling_space=True, layout="nhwc"):
    hw = [(2, get_edge_small()), (3, get_cloud_small())]
    levels = hw[hw_id][0]
    hw_config = acc.tileflow_accelerator_generator(
        hw[hw_id][1])
    dataflow = td.get_tileflow_conv_chain_dataflow(
        levels, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=define_tiling_space)

    with open(logfile, "r") as fin:
        work = False
        for line in fin:
            if "###############################################" in line:
                work = True
            if work:
                parts = line.split(',{')
                if len(parts) == 4:
                    first_parts = parts[0].split(",")
                    dataflow_, batch_, seq_len_, num_heads_, hidden_, metric_, seq_len__, hw_id_ = first_parts
                    second_parts = json.loads("{" + parts[1])
                    third_parts = json.loads(
                        ("{" + parts[2]).replace("'", '"'))
                    forth_parts = json.loads(
                        ("{" + parts[3]).replace("'", '"').replace("True", "1"))
                    if int(hw_id_) == hw_id and dataflow_name == dataflow_ and batch == int(batch_) and seq_len == int(seq_len_) and num_heads == int(num_heads_) and hidden == int(hidden_) and metric_ == metric_type:
                        perf, _, _ = inference(hw_config, dataflow, [], json.dumps(
                            second_parts), metric_type, resource_check=resource_check, debug=debug)
                        return perf

    raise RuntimeError("No such log entry found.")


def replay_config(config, dataflow_name, hw_id, batch, num_heads, seq_len, hidden, metric_type, debug=False, resource_check=True, define_tiling_space=True, layout="nhwc"):
    hw = [(2, get_edge_small()), (3, get_cloud_small())]
    levels = hw[hw_id][0]
    hw_config = acc.tileflow_accelerator_generator(
        hw[hw_id][1])
    dataflow = td.get_tileflow_conv_chain_dataflow(
        levels, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=define_tiling_space)

    print(config)
    perf, _, _ = inference(hw_config, dataflow, [],
                           config, metric_type, resource_check=resource_check, debug=debug)
    return perf


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
    parser.add_argument("--layout", type=str, default="nhwc")
    parser.add_argument("--inference", default=False, action="store_true")
    parser.add_argument("--logfile", type=str, default="trace.log",
                        help="Specify the logfile")
    parser.add_argument("--inference_hw", type=int,
                        help="The hw id to inference", default=0)
    parser.add_argument("--config_key", type=str, default="",
                        help="Manual Config Key")

    args = parser.parse_args()
    batch = args.batch
    trials = args.trials

    metric_type = args.metric
    if args.inference:
        shape = shapes[args.begin]
        num_heads = shape[0]
        seq_len = shape[1]
        hidden = shape[2]
        if args.config_key:
            perf = replay_config(args.config_key.replace("'", '"'), args.dataflow, args.inference_hw, batch, num_heads, seq_len,
                                 hidden, args.metric, args.debug, args.check_resource, args.define_tiling_space, args.layout)
        else:
            perf = replay(args.logfile, args.dataflow, args.inference_hw, batch, num_heads, seq_len,
                          hidden, args.metric, args.debug, args.check_resource, args.define_tiling_space, args.layout)
        print(perf)

    else:
        results_for_shape = []
        for shape in shapes[args.begin:args.begin+args.number]:
            results = []
            in_channel, height, width, out_channel_1, out_channel_2 = shape
            for i, (levels, hw) in enumerate(zip([2, 3], [get_edge_small(), get_cloud_small()])):
                print(
                    f"Current Task for metric={args.metric}, batch={args.batch}, height={height}, width={width}, in_channel={in_channel}, out_channel_1={out_channel_1}, out_channel_2={out_channel_2}, hw_id={i}")
                hw_config = acc.tileflow_accelerator_generator(hw)
                perf, key, config = run(
                    levels, hw_config, batch, height, width, in_channel, out_channel_1, out_channel_2, trials, metric_type=metric_type, debug=args.debug, resource_check=args.check_resource, define_tiling_space=args.define_tiling_space, layout=args.layout)
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
