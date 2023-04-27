from subprocess import Popen, PIPE
import os
import tempfile
import signal


def run_tileflow(workload, arch, mapping, tileflow_path="tileflow", save_tmp_file=False, unlimited_resource=False, tileflow_mapper_metric=None):
    tmpdir = tempfile.TemporaryDirectory()
    workload_file = os.path.join(tmpdir.name, "tmp_tileflow_workload.yaml")
    arch_file = os.path.join(tmpdir.name, "tmp_tileflow_arch_file.yaml")
    mapping_file = os.path.join(tmpdir.name, "tmp_tileflow_mapping_file.yaml")
    result_file = os.path.join(tmpdir.name, "tmp_tileflow_result_file.log")
    with open(workload_file, "w") as fout:
        fout.write(workload)
    with open(arch_file, "w") as fout:
        fout.write(arch)
    with open(mapping_file, "w") as fout:
        fout.write(mapping)

    option_file = os.path.join(
        tmpdir.name, "tmp_tileflow_option_file.yaml")

    objective = "energy" if tileflow_mapper_metric is not None and "energy" in tileflow_mapper_metric else "cycle"
    with open(option_file, "w") as fout:
        fout.write("check:\n")
        fout.write(f"  mem: {not unlimited_resource}\n")
        fout.write(f"  spatial: {not unlimited_resource}\n")

        fout.write(f"tileflow-mapper:\n")
        fout.write(f"  objective: {objective}")
    # with open(option_file, "r") as fin:
    #     for line in fin:
    #         print(line, flush=True)
    command = [
        tileflow_path,
        workload_file,
        arch_file,
        mapping_file,
        option_file
    ]

    process = Popen(command, stdout=PIPE, stdin=PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    def singal_handler(signal, frame):
        os.killpg(os.getpgid(process.pid), 9)

    signal.signal(signal.SIGINT, singal_handler)

    with open(result_file, "w") as fout:
        fout.write(stdout.decode())
    parse_results = {
        "Cycle": int,
        "Energy": float,
        "mac::Flops": float,
        "mac::Energy": float,
    }
    for name in ["MEM", "SPATIAL"]:
        for level in ["L0", "L1", "L2", "L3"]:
            parse_results[f"{name}::{level}"] = float
    for level in ["L0", "L1", "L2", "L3"]:
        for name in ["Accesses", "Read", "Fill", "Update", "CapUtil", "SpatialUtil", "SlowDown", "Energy"]:
            parse_results[f"{level}::{name}"] = float
    results = {}
    in_results = False
    try:
        with open(result_file, "r") as fin:
            for line in fin:
                if "***TileFlow Result" in line:
                    in_results = True
                if "***TileFlow Result Ends" in line:
                    in_results = False
                if in_results:
                    parts = line.split(",")
                    if parts[0] in parse_results:
                        if parts[0] in results:
                            results[parts[0]] = max(
                                results[parts[0]], parse_results[parts[0]](parts[1]))
                        else:
                            results[parts[0]] = parse_results[parts[0]](
                                parts[1])
        if "Cycle" in results and "Energy" in results:
            if results["Cycle"] == 0:
                results["status_ok"] = False
            else:
                results["status_ok"] = True
        else:
            results["status_ok"] = False
    except Exception as e:
        results["status_ok"] = False

    # if not save_tmp_file:
    #     os.remove(workload_file)
    #     os.remove(arch_file)
    #     os.remove(mapping_file)
        # if os.path.exists(result_file) and os.path.isfile(result_file):
        #     os.remove(result_file)

    return results
