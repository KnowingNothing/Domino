import os
from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
from collections import namedtuple


MAESTRO_PATH_ENV = "MAESTRO_PATH"


def find_maestro():
    maestro_path = os.getenv(MAESTRO_PATH_ENV, None)
    found = False
    if maestro_path is None:
        # some dirty trick to find maestro
        possible_paths = ["../../../maestro", "../../maestro", "../maestro",
                          "./maestro"]
        for pp in possible_paths:
            maestro_path = pp
            if os.path.exists(maestro_path):
                maestro_path = os.path.join(maestro_path, "maestro")
                if os.path.exists(maestro_path) and os.path.isfile(maestro_path):
                    found = True
                    break
    if not found or maestro_path is None:
        raise RuntimeError(
            "Please set MAESTRO_PATH to the position of Maestro executable")
    return os.path.abspath(maestro_path)


def generate_maestro_command(
    maestro_path,
    mapping_file_name,
    noc_bw,
    off_chip_bw,
    num_pes,
    l1_size,
    l2_size,
    noc_hops=1,
    noc_hop_latency=1,
    noc_mc_support=True,
    num_simd_lanes=1,
):

    command = [maestro_path,
               f"--Mapping_file=./{mapping_file_name}.m",
               f"--full_buffer=false",
               f"--noc_bw_cstr={noc_bw}",
               f"--noc_hops={noc_hops}",
               f"--noc_hop_latency={noc_hop_latency}",
               f"--offchip_bw_cstr={off_chip_bw}",
               f"--noc_mc_support={'true' if noc_mc_support else 'false'}",
               f"--num_pes={num_pes}",
               f"--num_simd_lanes={num_simd_lanes}",
               f"--l1_size_cstr={l1_size}",
               f"--l2_size_cstr={l2_size}",
               "--print_res=false",
               "--print_res_csv_file=true",
               "--print_log_file=false",
               "--print_design_space=false",
               "--msg_print_lv=0"]

    return command


MaestroResults = namedtuple(
    "MaestroResults",
    [
        "runtime",
        "runtime_series",
        "throughput",
        "energy",
        "area",
        "power",
        "l1_size",
        "l2_size",
        "l1_size_series",
        "l2_size_series",
        "l1_input_read",
        "l1_weight_read",
        "l1_output_read",
        "l1_input_write",
        "l1_weight_write",
        "l1_output_write",
        "l2_input_read",
        "l2_weight_read",
        "l2_output_read",
        "l2_input_write",
        "l2_weight_write",
        "l2_output_write",
        "mac"
    ]
)


def run_maestro(mapping_file_name, command):
    process = Popen(command, stdout=PIPE, stdin=PIPE)
    stdout, stderr = process.communicate()
    process.wait()
    
    try:
        df = pd.read_csv(f"./{mapping_file_name}.csv")
        layer_name = df[" Layer Number"]
        runtime = np.array(df[" Runtime (Cycles)"])
        runtime_series = np.array(df[" Runtime (Cycles)"])
        throughput = np.array(df[" Throughput (MACs/Cycle)"])
        energy = np.array(
            df[" Activity count-based Energy (nJ)"])
        area = np.array(df[" Area"])
        power = np.array(df[" Power"])
        l1_size = np.array(df[" L1 SRAM Size Req (Bytes)"])
        l2_size = np.array(df["  L2 SRAM Size Req (Bytes)"])
        l1_size_series = np.array(
            df[" L1 SRAM Size Req (Bytes)"])
        l2_size_series = np.array(
            df["  L2 SRAM Size Req (Bytes)"])
        l1_input_read = np.array(df[" input l1 read"])
        l1_input_write = np.array(df[" input l1 write"])
        l1_weight_read = np.array(df["filter l1 read"])
        l1_weight_write = np.array(df[" filter l1 write"])
        l1_output_read = np.array(df["output l1 read"])
        l1_output_write = np.array(df[" output l1 write"])
        l2_input_read = np.array(df[" input l2 read"])
        l2_input_write = np.array(df[" input l2 write"])
        l2_weight_read = np.array(df[" filter l2 read"])
        l2_weight_write = np.array(df[" filter l2 write"])
        l2_output_read = np.array(df[" output l2 read"])
        l2_output_write = np.array(df[" output l2 write"])
        mac = np.array(df[" Num MACs"])
        if os.path.exists(f"{mapping_file_name}.csv") and os.path.isfile(f"{mapping_file_name}.csv"):
            os.remove(f"{mapping_file_name}.csv")
        return MaestroResults(
            runtime, runtime_series,
            throughput, energy, area,
            power, l1_size, l2_size, l1_size_series,
            l2_size_series, l1_input_read, l1_weight_read,
            l1_output_read, l1_input_write, l1_weight_write,
            l1_output_write, l2_input_read, l2_weight_read,
            l2_output_read, l2_input_write, l2_weight_write,
            l2_output_write, mac
        )
    except Exception as e:
        print(stdout)
        with open(f"./{mapping_file_name}.m", "r") as fin:
            for line in fin:
                print(line, end="")
        raise e


if __name__ == "__main__":
    maestro_path = find_maestro()
    print(maestro_path)
