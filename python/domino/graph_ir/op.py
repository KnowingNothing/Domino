from optparse import Option
from typing import List, Dict, Optional, Union
import enum
import numpy as np
from ..type_system import GeneralDType
from .op_base import OpBase
from .tensor import Tensor, Attribute
from ..program_ir import ConstFloat, ConstInt, ConstString
from .quantize import OpQuantParam, TensorQuantParam


__all__ = ["OpName", "NamedOp", "ConvOp", "ActivationAttr"]


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
        Clip = "Clip"

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
                
    def __str__(self):
        return f"{self.name}(\n\t{self.inputs},\n\t{self.outputs})"
    
    
    def __repr__(self) -> str:
        return f"{self.name}(\n\t{self.inputs},\n\t{self.outputs})"


GeneralInt = Union[int, ConstInt]


class ConvOp(NamedOp):
    def __init__(
        self,
        inputs: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        strides: Union[GeneralInt, List[GeneralInt]] = 1,
        padding: Union[GeneralInt, List[GeneralInt]] = 0,
        dilation: Union[GeneralInt, List[GeneralInt]] = 1,
        output_dtype: GeneralDType = "float32",
        output_layout: Optional[str] = "NCHW",
        output_tensor_idx: Union[int, str] = "",
        output_quant_params: Optional[TensorQuantParam] = None,
        conv_quant_params: Optional[OpQuantParam] = None,
        attrs: Optional[Dict[str, Attribute]] = None
    ) -> None:
        super(ConvOp, self).__init__(
            OpName.ConvOp.Conv2d, {}, {}, conv_quant_params, attrs)
        self.inputs = {"inputs": inputs, "weight": weight, "bias": bias} if bias is not None else {
            "inputs": inputs, "weight": weight}
        self.use_bias = bias is not None
        self.strides = [strides, strides] if isinstance(
            strides, (int, ConstInt)) else strides
        assert isinstance(self.strides, (tuple, list)
                          ) and len(self.strides) == 2
        self.padding = [padding, padding] if isinstance(
            padding, (int, ConstInt)) else padding
        assert isinstance(self.padding, (tuple, list)
                          ) and len(self.padding) == 2
        self.dilation = [dilation, dilation] if isinstance(
            dilation, (int, ConstInt)) else dilation
        assert isinstance(self.padding, (tuple, list)
                          ) and len(self.padding) == 2

        assert inputs.layout is not None
        assert weight.layout is not None

        all_dim_shape_dict = {}
        for tensor in [inputs, weight]:
            assert tensor.layout is not None, f"{tensor} has no layout information."
            for k, v in zip(tensor.shape, tensor.layout):
                if k in all_dim_shape_dict:
                    assert v == all_dim_shape_dict[
                        k], f"Shape at dim {k} mismatches: {all_dim_shape_dict[k]} vs {v}"
                else:
                    all_dim_shape_dict[k] = v

        H = all_dim_shape_dict["H"]
        W = all_dim_shape_dict["W"]
        R = all_dim_shape_dict["R"]
        S = all_dim_shape_dict["S"]
        kR = (R - 1) * self.dilation[0] + 1
        kS = (S - 1) * self.dilation[1] + 1
        P = (H + 2 * self.padding[0] - kR) // self.strides[0] + 1
        Q = (W + 2 * self.padding[1] - kS) // self.strides[1] + 1
        all_dim_shape_dict["P"] = P
        all_dim_shape_dict["Q"] = Q

        output_shape = []
        output_layout = output_layout.replace("H", "P").replace("w", "Q")
        for k in output_layout:
            if k not in all_dim_shape_dict:
                raise RuntimeError(
                    f"Dimension {k} specified in output layout {output_layout} is not found.")
            output_shape.append(all_dim_shape_dict[k])

        output_key = "output"

        output_tensor = Tensor(
            output_shape,
            output_dtype,
            out_idx=output_key,
            layout=output_layout,
            name="conv2d_output",
            quant_params=output_quant_params,
            tensor_idx=output_tensor_idx
        )

        self.outputs = {
            output_key: output_tensor
        }
