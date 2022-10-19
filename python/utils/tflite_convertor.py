"""This file is adapted from tinyengine: 

    https://github.com/mit-han-lab/tinyengine/blob/master/code_generator/TfliteConvertor.py

"""


from typing import Dict
from .tflite import Model
from .tflite.BuiltinOperator import BuiltinOperator


def load_tflite_model(path: str) -> Model.Model:
    with open(path, "rb") as fin:
        return Model.Model.GetRootAsModel(fin, 0)


def get_tflite_builtin_op_map(builtins: BuiltinOperator) -> Dict[int, str]:
    op_map = {}
    for k, v in builtins.__dict__.items():
        if k.upper() == k and not k.startswith("_") and isinstance(v, int):
            op_map[v] = k
    return op_map


class TfliteOpConvertor(object):
    pass


class Conv2dConvertor(TfliteOpConvertor):
    def convert(self):
        pass


class AddConvertor(TfliteOpConvertor):
    pass


class AveragePool2dConvertor(TfliteOpConvertor):
    pass


class DepthwiseConv2dConvertor(TfliteOpConvertor):
    pass


class PadConvertor(TfliteOpConvertor):
    pass


class ResizeNearestNeighborConvertor(TfliteOpConvertor):
    pass


class MaxPool2dConvertor(TfliteOpConvertor):
    pass


class MeanConvertor(TfliteOpConvertor):
    pass


class TransposeConvertor(TfliteOpConvertor):
    pass


class FullyConnectedConvertor(TfliteOpConvertor):
    pass


CONVERT_MAP = {
    "CONV_2D": Conv2dConvertor,
    "ADD": AddConvertor,
    "AVERAGE_POOL_2D": AveragePool2dConvertor,
    "DEPTHWISE_CONV_2D": DepthwiseConv2dConvertor,
    "PAD": PadConvertor,
    "RESIZE_NEAREST_NEIGHBOR": ResizeNearestNeighborConvertor,
    "MAX_POOL_2D": MaxPool2dConvertor,
    "MEAN": MeanConvertor,
    "TRANSPOSE": TransposeConvertor,
    "FULLY_CONNECTED": FullyConnectedConvertor,
}


class TfliteConvertor(object):
    def __init__(self, path: str) -> None:
        self.model_path = path

    def parse(self):
        model = load_tflite_model(self.model_path)
        subgraph = model.Subgraphs(0)
        num_ops = subgraph.OperatorsLength()
        op_map = get_tflite_builtin_op_map(BuiltinOperator())

        for i in range(num_ops):
            op = subgraph.Operators(i)

            self.parse_op(model, subgraph, op)

    def parse_op(self, model, subgraph, op_map, op):
        op_code = op.OpcodeIndex()
        op_code_id = model.OperatorCodes(op_code).DeprecatedBuiltinCode()
        op_name = op_map[op_code_id]
