
from tileflow import (get_edge_small, get_cloud_small,
                      get_edge_large, get_cloud_large, tuning, inference)
import tileflow.dataflows as td

import domino.accelerator as acc
import argparse
import json


def run(dataflow_name, levels, hw_config, batch, height, width, in_channel, out_channel_1, out_channel_2, trials, metric_type, debug=False, resource_check=True, define_tiling_space=False, layout="nhwc", second_kernel_size=3):
    table = {
        "fused_layer": td.get_fused_layer_dataflow,
        "isos": td.get_isos_dataflow,
        "tileflow": td.get_tileflow_dataflow,
        "no_fuse": td.get_conv_chain_no_fuse_dataflow
    }
    dataflow = table[dataflow_name](
        levels, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=define_tiling_space, layout=layout, second_kernel_size=second_kernel_size)

    best_perf, best_config_key, best_config = tuning(
        hw_config, dataflow, [], trials, metric_type, sequential=debug, debug=debug, resource_check=resource_check)
    return best_perf, best_config_key, best_config


def replay(dataflow_name, l1_bw, logfile, hw_id, batch, height, width, in_channel, out_channel_1, out_channel_2, metric_type, debug=False, resource_check=True, define_tiling_space=True, layout="nhwc", second_kernel_size=3):
    hw = [(2, get_edge_small(L1_BW=l1_bw)), (3, get_cloud_small(L1_BW=l1_bw))]
    levels = hw[hw_id][0]
    hw_config = acc.tileflow_accelerator_generator(
        hw[hw_id][1])
    table = {
        "fused_layer": td.get_fused_layer_dataflow,
        "isos": td.get_isos_dataflow,
        "tileflow": td.get_tileflow_dataflow,
        "no_fuse": td.get_conv_chain_no_fuse_dataflow
    }
    dataflow = table[dataflow_name](
        levels, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=define_tiling_space, layout=layout, second_kernel_size=second_kernel_size)

    with open(logfile, "r") as fin:
        work = False
        for line in fin:
            if "###############################################" in line:
                work = True
            if work:
                parts = line.split(',{')
                if len(parts) == 4:
                    first_parts = parts[0].split(",")
                    batch_, height_, width_, in_channel_, out_channel_1_, out_channel_2_, metric_, hw_id_ = first_parts
                    # print(first_parts)
                    # print(batch, height, width, in_channel,
                    #       out_channel_1, out_channel_2, metric_)
                    second_parts = json.loads("{" + parts[1])
                    third_parts = json.loads(
                        ("{" + parts[2]).replace("'", '"'))
                    forth_parts = json.loads(
                        ("{" + parts[3]).replace("'", '"').replace("True", "1"))
                    if int(hw_id_) == hw_id and batch == int(batch_) and height == int(height_) and width == int(width_) and in_channel == int(in_channel_) and out_channel_1 == int(out_channel_1_) and out_channel_2 == int(out_channel_2_):
                        perf, _, _ = inference(hw_config, dataflow, [], json.dumps(
                            second_parts), metric_type, resource_check=resource_check, debug=debug)
                        return perf

    raise RuntimeError("No such log entry found.")


def replay_config(dataflow_name, l1_bw, config, hw_id, batch, height, width, in_channel, out_channel_1, out_channel_2, metric_type, debug=False, resource_check=True, define_tiling_space=True, layout="nhwc", second_kernel_size=3):
    hw = [(2, get_edge_small(L1_BW=l1_bw)), (3, get_cloud_small(L1_BW=l1_bw))]
    levels = hw[hw_id][0]
    hw_config = acc.tileflow_accelerator_generator(
        hw[hw_id][1])
    table = {
        "fused_layer": td.get_fused_layer_dataflow,
        "isos": td.get_isos_dataflow,
        "tileflow": td.get_tileflow_dataflow,
        "no_fuse": td.get_conv_chain_no_fuse_dataflow
    }
    dataflow = table[dataflow_name](
        levels, batch, height, width, in_channel, out_channel_1, out_channel_2, define_tiling_space=define_tiling_space, layout=layout, second_kernel_size=second_kernel_size)

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
    (64, 56, 56, 64, 64),
    (64, 56, 56, 128, 128),
    (256, 56, 56, 256, 64)
]


"""example
python fused_layer_dataflow.py --metric=1e9/latency --trials 1 |& tee trace.log
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric", type=str, help="Evaluation metric type [1e9/latency, 1e9/energy, Utilization_L0, Utilization_L1, Utilization_L2, Utilization_L3]", default="1e9/latency")
    parser.add_argument("--dataflow", type=str,
                        help="Conv dataflow", default="fused_layer")
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
    parser.add_argument("--second_kernel_size", type=int,
                        help="Kernel size of the second convolution", default=1)
    parser.add_argument("--l1_bw", type=int,
                        help="The L1 BW lower bound", default=100)
    parser.add_argument("--l2_bw", type=int,
                        help="The L1 BW lower bound", default=10)

    args = parser.parse_args()
    batch = args.batch
    trials = args.trials

    metric_type = args.metric

    if args.inference:
        shape = shapes[args.begin]
        in_channel, height, width, out_channel_1, out_channel_2 = shape
        results = []
        if args.config_key:
            for l1_bw in range(args.l1_bw, 500, 10):
                perf = replay_config(args.dataflow, l1_bw, args.config_key.replace("'", '"'), args.inference_hw, batch, height, width, in_channel, out_channel_1, out_channel_2,
                                     args.metric, args.debug, args.check_resource, args.define_tiling_space, args.layout, second_kernel_size=args.second_kernel_size)
                perf = perf[args.metric]
                results.append((l1_bw, perf))
        else:
            for l1_bw in range(args.l1_bw, 500, 10):
                perf = replay(args.dataflow, l1_bw, args.logfile, args.inference_hw, batch, height, width, in_channel, out_channel_1, out_channel_2,
                              args.metric, args.debug, args.check_resource, args.define_tiling_space, args.layout, second_kernel_size=args.second_kernel_size)
                perf = perf[args.metric]
                results.append((l1_bw, perf))

        with open("l1_bw_result.csv", "w") as fout:
            fout.write(f"L1_BW,{args.metric}\n")
            for res in results:
                fout.write(f"{res[0]},{res[1]}\n")

    else:
        hws = [(2, get_edge_small), (3, get_cloud_small)]
        results_for_shape = []
        for shape in shapes[args.begin:args.begin+args.number]:
            results = []
            in_channel, height, width, out_channel_1, out_channel_2 = shape
            levels = hws[args.inference_hw][0]
            hw_func = hws[args.inference_hw][1]
            i = args.hw_id
            for l1_bw in range(args.l1_bw, 500, 10):
                print(
                    f"Current Task for metric={args.metric}, batch={args.batch}, height={height}, width={width}, in_channel={in_channel}, out_channel_1={out_channel_1}, out_channel_2={out_channel_2}, hw_id={i}")
                hw_config = acc.tileflow_accelerator_generator(
                    hw_func(L1_BW=l1_bw))
                perf, key, config = run(
                    args.dataflow, levels, hw_config, batch, height, width, in_channel, out_channel_1, out_channel_2, trials, metric_type=metric_type, debug=args.debug, resource_check=args.check_resource, define_tiling_space=args.define_tiling_space, layout=args.layout, second_kernel_size=args.second_kernel_size)
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
