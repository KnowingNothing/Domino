import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="data_log.log")

args = parser.parse_args()


results = []

title = ""
with open(args.file, "r") as fin:
    work = False
    for line in fin:
        if "Temporal Best Score: " in line and work:
            _, value = line.split(": ")
            value = float(value)
            for i in range(5):
                results.append(value)
        if "Current Task for" in line:
            if "hw_id=0" in line:
                work = True
                results.append("\n")
            else:
                work = False
           
with open("tmp.log", "w") as fout:
    for res in results:
        fout.write(f"{res}\n")