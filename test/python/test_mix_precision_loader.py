import json
import os
from domino.graph_ir import Op
from domino.graph_pass import set_graph_precision, GraphPrinter
from domino.utils import ONNXConvertor


def get_precision_configs(path: str):
    if not (os.path.exists(path) and os.path.isfile(path)):
        raise RuntimeError(f"The file {path} is not found")

    ret = []

    with open(path, "r") as fin:
        for line in fin:
            config = {}
            obj = json.loads(line)
            # acc = obj["acc"]
            # cost = obj["tot_cost"]
            layers = obj["layers"]
            for tid, conf in layers.items():
                if 'o_bit' in conf:
                    config[tid] = f"int{conf['o_bit']}"
                elif 'w_bit' in conf:
                    config[tid] = f"int{conf['w_bit']}"
                else:
                    raise ValueError()
            ret.append(config)

    return ret


def get_graph(path: str):
    convertor = ONNXConvertor(path, inference=True)
    graph = convertor.parse()
    return graph


def test_basic_set_precision():
    config_path = "new_resnet18_pareto.json"
    model_path = "raw_resnet18.onnx"

    graph = get_graph(model_path)
    configs = get_precision_configs(config_path)

    graphs = []
    for config in configs:
        new_graph = set_graph_precision(
            graph,
            config,
            graph_inputs_precision="int8",
            target_ops=[
                *Op.all_ops_in(Op.OpName.ActivationOp),
                *Op.all_ops_in(Op.OpName.PoolingOp),
                *Op.all_ops_in(Op.OpName.ScalingOp),
                *Op.all_ops_in(Op.OpName.ElementwiseOp),
                *Op.all_ops_in(Op.OpName.PadOp),
                *Op.all_ops_in(Op.OpName.ReduceOp),
                *Op.all_ops_in(Op.OpName.DimOrderOp), ]
        )
        graphs.append(new_graph)

    printer = GraphPrinter()
    ret = printer(graphs[0])
    # print(ret)
    assert ret


if __name__ == "__main__":
    test_basic_set_precision()
