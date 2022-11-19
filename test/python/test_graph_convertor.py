import json
import os
import networkx as nx
from typing import Set, Optional, Union, List, Any
from domino import utils
from domino.graph_ir import Op, Tensor, Graph
from domino.graph_pass import set_graph_precision, GraphPrinter, GraphVisitor
from domino.utils import ONNXConvertor



def get_graph(path: str):
    convertor = ONNXConvertor(path, inference=True)
    graph = convertor.parse()
    return graph


def test_convertor():
    # config_path = "new_resnet18_pareto.json"
    # model_path = "raw_resnet18.onnx"
    # config_path = "new_mobilenetv2_pareto.json"
    # model_path = "raw_mobilenetv2.onnx"
    # config_path = "new_resnet50_pareto.json"
    # model_path = "raw_resnet50.onnx"
    # config_path = "w_a_joint_yolov5_pareto.json"
    # model_path = "yolov5s_640x640.simplify.onnx"
    # model_path = "googlenet-12.onnx"
    # model_path = "unet_13_256.onnx"
    model_path = "simplified_bert_base.onnx"

    graph = get_graph(model_path)
    


if __name__ == "__main__":
    test_convertor()
