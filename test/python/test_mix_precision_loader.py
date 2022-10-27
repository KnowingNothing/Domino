import json
import os
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
                config[tid] = f"int{conf['o_bit']}"
            ret.append(config)

    return ret


def get_graph(path: str):
    convertor = ONNXConvertor(path, inference=True)
    graph = convertor.parse()
    return graph


def test_basic_set_precision():
    config_path = "raw_resnet18.json"
    model_path = "raw_resnet18.onnx"

    graph = get_graph(model_path)
    configs = get_precision_configs(config_path)

    graphs = []
    for config in configs:
        new_graph = set_graph_precision(graph, config)
        graphs.append(new_graph)

    printer = GraphPrinter()
    ret = printer(graphs[0])
    assert ret


if __name__ == "__main__":
    test_basic_set_precision()
