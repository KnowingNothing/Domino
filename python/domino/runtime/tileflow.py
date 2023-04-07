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

    if not save_tmp_file:
        os.remove(workload_file)
        os.remove(arch_file)
        os.remove(mapping_file)
        # if os.path.exists(result_file) and os.path.isfile(result_file):
        #     os.remove(result_file)
