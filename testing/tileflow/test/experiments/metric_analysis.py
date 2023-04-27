import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="data_log.log")
parser.add_argument("--hw_id", type=str, default="0")
parser.add_argument("--metric", type=str, default="Cycle", choices=["Cycle", "Energy", "MEM::L1", "MEM::L2", "MEM::L0", "all"])
parser.add_argument("--wkl", type=str, default="self_attention")

args = parser.parse_args()


results = []

title = ""
with open(args.file, "r") as fin:
    work = False
    for line in fin:
        if "###############################################" in line:
            work = True
        # dataflow,batch,seq_len,num_heads,hidden,metric,seq_len,hw_id,key,config,perf
        # 1,112,112,64,192,128,1e9/latency,0
        # rgran,1,512,8,512,1e9/latency,512,0,{"value": null, "children": {"M": {"value": 20, "children": {}}, "L": {"value": 6, "children": {}}, "B": {"value": 0, "children": {}}, "H": {"value": 9, "children": {}}, "N": {"value": 1, "children": {}}, "A": {"value": 1, "children": {}}}},{'M': [4, 2, 64], 'L': [64, 8], 'B': [1, 1, 1], 'H': [8, 1, 1], 'N': [2, 32], 'A': [2, 32]},{'Cycle': 917504, 'Energy': 20578400000.0, 'MEM::L0': 1.0, 'MEM::L1': 0.048, 'MEM::L2': 5.13e-06, 'status_ok': True}
        if work:
            parts = line.split(',{')
            if len(parts) == 4:
                first_parts = parts[0].split(",")
                if args.wkl == "self_attention":
                    if len(first_parts) == 8:
                        dataflow, batch, seq_len, num_heads, hidden, metric, seq_len, hw_id = first_parts
                    elif len(first_parts) == 7:
                        batch, seq_len, num_heads, hidden, metric, seq_len, hw_id = first_parts
                    else:
                        raise RuntimeError()
                else:
                    batch, height, width, in_channel, out_channel_1, out_channel_2, metric, hw_id = first_parts
                  
                second_parts = json.loads("{" + parts[1])
                third_parts = json.loads(("{" + parts[2]).replace("'", '"'))
                forth_parts = json.loads(("{" + parts[3]).replace("'", '"').replace("True", "1"))
                if hw_id == args.hw_id:
                    if args.metric in forth_parts:
                        if args.wkl == "self_attention":
                            results.append(f"{batch},{seq_len},{num_heads},{hidden},{metric},{hw_id},{forth_parts[args.metric]}")
                        else:
                            results.append(f"{batch},{height},{width},{in_channel},{out_channel_1},{out_channel_2},{metric},{hw_id},{forth_parts[args.metric]}")
                    elif args.metric == "all":
                        s = ""
                        title = ""
                        for k, v in forth_parts.items():
                            title += "," + k
                            s += f",{v}"
                        if args.wkl == "self_attention":
                            results.append(f"{batch},{seq_len},{num_heads},{hidden},{metric},{hw_id}{s}")
                        else:
                            results.append(f"{batch},{height},{width},{in_channel},{out_channel_1},{out_channel_2},{metric},{hw_id}{s}")

if args.metric == "all":
    if args.wkl == "self_attention":
        print(f"batch,seq_len,num_heads,hidden,metric,hw_id{title}")       
    else:
        print(f"batch,height,width,in_channel,out_channel_1,out_channel_2,metric,hw_id{title}")  
else:
    if args.wkl == "self_attention":
        print(f"batch,seq_len,num_heads,hidden,metric,hw_id,{args.metric}")       
    else:
        print(f"batch,height,width,in_channel,out_channel_1,out_channel_2,metric,hw_id,{args.metric}")  
for res in results:
    print(res)