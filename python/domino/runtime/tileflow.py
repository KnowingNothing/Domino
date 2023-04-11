from subprocess import Popen, PIPE
import os


def run_tileflow(workload, arch, mapping, tileflow_path="tileflow", save_tmp_file=False):
    workload_file = "tmp_tileflow_workload.yaml"
    arch_file = "tmp_tileflow_arch_file.yaml"
    mapping_file = "tmp_tileflow_mapping_file.yaml"
    result_file = "tmp_tileflow_result_file.log"
    with open(workload_file, "w") as fout:
        fout.write(workload)
    with open(arch_file, "w") as fout:
        fout.write(arch)
    with open(mapping_file, "w") as fout:
        fout.write(mapping)
    command = [
        tileflow_path,
        workload_file,
        arch_file,
        mapping_file,
    ]

    process = Popen(command, stdout=PIPE, stdin=PIPE)
    stdout, stderr = process.communicate()
    process.wait()
    with open(result_file, "w") as fout:
        fout.write(stdout.decode())
    parse_results = {
        "Cycle": int,
        "Energy": float
    }
    results = {}
    in_results = False
    with open(result_file, "r") as fin:
        for line in fin:
            if line.startswith("***TileFlow Result"):
                in_results = True
            if line.startswith("***TileFlow Result Ends"):
                in_results = False
            if in_results:
                parts = line.split(",")
                print(parts)
                if parts[0] in parse_results:
                    results[parts[0]] = parse_results[parts[0]](parts[1])

    if not save_tmp_file:
        os.remove(workload_file)
        os.remove(arch_file)
        os.remove(mapping_file)
        # if os.path.exists(result_file) and os.path.isfile(result_file):
        #     os.remove(result_file)
    
    return results
