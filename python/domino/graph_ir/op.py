from typing import List, Dict, Optional
import enum
import numpy as np
from .op_base import OpBase
from .tensor import Tensor, Attribute
from ..program_ir import ConstFloat, ConstInt, ConstString
from .quantize import OpQuantParam


class OpName(object):
    class ConvOp(enum.Enum):
        Conv2d = "Conv2d"
        Conv2dBias = "Conv2dBias"
        Conv2dClip = "Conv2dClip"
        Conv2dReLU = "Conv2dReLU"
        Conv2dBiasClip = "Conv2dBiasClip"
        Conv2dBiasReLU = "Conv2dBiasReLU"
        DepthwiseConv2d = "DepthwiseConv2d"

    class MatrixOp(enum.Enum):
        FullyConnected = "FullyConnected"
        Gemm = "Gemm"

    class PadOp(enum.Enum):
        Pad = "Pad"

    class ElementwiseOp(enum.Enum):
        Add = "ElemAdd"
    
    class ActivationOp(enum.Enum):
        ReLU = "ReLU"

    class PoolingOp(enum.Enum):
        AveragePool1d = "AveragePool1d"
        AveragePool2d = "AveragePool2d"
        AveragePool3d = "AveragePool3d"
        AveragePool = "AveragePool"
        MaxPool1d = "MaxPool1d"
        MaxPool2d = "MaxPool2d"
        MaxPool3d = "MaxPool3d"
        MaxPool = "MaxPool"
        GlobalAveragePool1d = "GlobalAveragePool1d"
        GlobalAveragePool2d = "GlobalAveragePool2d"
        GlobalAveragePool3d = "GlobalAveragePool3d"
        GlobalAveragePool = "GlobalAveragePool"

    class ScalingOp(enum.Enum):
        ResizeNearestNeighbor = "ResizeNearestNeighbor"
        Reshape = "Reshape"
        Flatten = "Flatten"

    class ReduceOp(enum.Enum):
        Mean = "Mean"

    class DimOrderOp(enum.Enum):
        Transpose = "Transpose"
        
        
class ActivationAttr(Attribute):
    def __init__(self, value: str) -> None:
        super(ActivationAttr, self).__init__(ConstString(value))


class NamedOp(OpBase):
    def __init__(
        self,
        name: enum.Enum,
        inputs: Dict[str, Tensor],
        outputs: Dict[str, Tensor],
        quant_params: Optional[OpQuantParam] = None,
        attrs: Optional[Dict[str, Attribute]] = None
    ) -> None:
        """NamedOp

        Operator defined by name. e.g., conv2d, gemm

        Args:
            name (str): the name of the operator
            inputs (Dict[str, Tensor]): the inputs of the operator
            outputs (Dict[str, Tensor]): the outputs of the operator
            quant_params (Optional[OpQuantParam]): operator quantize params
            attrs (Optional[Dict[str, Attribute]]): the attributes of the operator
        """
        super(NamedOp, self).__init__()
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.quant_params = quant_params
        self.attrs = attrs
        
        # set the produce op and out_idx
        for k in self.outputs.keys():
            self.outputs[k].produce_op = self
            self.outputs[k].out_idx = k
            
        # check attrs
        if attrs is not None:
            for k, v in attrs.items():
                assert isinstance(v, Attribute)
