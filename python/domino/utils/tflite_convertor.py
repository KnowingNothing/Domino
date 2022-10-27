

from sys import displayhook
from typing import Dict, Optional
from webbrowser import Opera
import numpy as np
import domino
from domino.graph_ir import Tensor, ConstTensor, Attribute, ActivationAttr, Op, SubGraph, Graph
from domino.program_ir import ConstInt, ConstFloat, ExprList, make_const
from domino.type_system import DType, ShapeType, GeneralDType
from domino.graph_ir.quantize import ClipQuantParam, QuantParam, ScaleQuantParam, TensorQuantParam


try:
    import tflite
    import tflite.BuiltinOperator
    import tflite.SubGraph
    from tflite.Operator import Operator
    from tflite.BuiltinOptions import BuiltinOptions
    # for conv2d
    from tflite.Conv2DOptions import Conv2DOptions
    from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
    from tflite.Padding import Padding
    from tflite.TensorType import TensorType
    from tflite.ActivationFunctionType import ActivationFunctionType
    # for elementwise
    from tflite.AddOptions import AddOptions
    from tflite.DivOptions import DivOptions
    from tflite.MulOptions import MulOptions
    from tflite.SubOptions import SubOptions
    # for pool2d
    from tflite.Pool2DOptions import Pool2DOptions
    # for reshape
    from tflite.ReshapeOptions import ReshapeOptions
except ImportError:
    raise ImportError("Can't import tflite package.\nTry `pip install tflite`")

try:
    import tflite.Model.Model as TFModel
except ImportError:
    TFModel = tflite.Model


def load_tflite_model(path: str):
    with open(path, "rb") as fin:
        buf = fin.read()
        return TFModel.GetRootAsModel(buf, 0)


def get_tflite_builtin_op_map(builtins: tflite.BuiltinOperator) -> Dict[int, str]:
    op_map = {}
    for k in builtins.__dir__():
        v = getattr(builtins, k)
        if k.upper() == k and not k.startswith("_") and isinstance(v, int):
            op_map[v] = k
    return op_map


def get_tflite_tensor_name(subgraph, tensor_idx):
    return subgraph.Tensors(tensor_idx).Name().decode("utf-8")


def _decode_type(n):
    _tflite_m = {
        0: "float32",
        1: "float16",
        2: "int32",
        3: "uint8",
        4: "int64",
        5: "string",
        6: "bool",
        7: "int16",
        8: "complex64",
        9: "int8",
    }
    return _tflite_m[n]


class TfliteTensor(object):
    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None) -> None:
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params

    @property
    def shape(self):
        return tuple([int(x) for x in self.tensor.ShapeAsNumpy()])

    @property
    def dtype(self):
        return _decode_type(self.tensor.Type())

    @property
    def value(self):
        data = self.buffer.DataAsNumpy()
        return np.frombuffer(data, dtype=self.dtype).reshape(self.shape)


def get_tflite_tensor(model, subgraph, tensor_idx):
    tensor = subgraph.Tensors(tensor_idx)
    buffer_idx = tensor.Buffer()
    buffer = model.Buffers(buffer_idx)

    qnn_params = None
    tflite_qnn_params = tensor.Quantization()
    if tflite_qnn_params is not None:
        tflite_scale = tflite_qnn_params.ScaleAsNumpy()
        tflite_zero_point = tflite_qnn_params.ZeroPointAsNumpy()
        is_qnn_params_valid = True

        if isinstance(tflite_scale, np.ndarray):
            assert isinstance(tflite_zero_point, np.ndarray)

            if tflite_scale.size != 1 and tflite_zero_point.size != 1:
                scale = tflite_scale
                zero_point = tflite_zero_point
                if not np.all(zero_point == 0):
                    raise RuntimeError(
                        "Invalid zero points for quantization")
                zero_point = int(zero_point[0])

            elif tflite_scale.size == 1 and tflite_zero_point.size == 1:
                scale = float(tflite_scale[0])
                zero_point = int(tflite_zero_point[0])

            else:
                raise NotImplementedError()

        elif tflite_scale == 0 and tflite_zero_point == 0:
            is_qnn_params_valid = False
            scale = 0
            zero_point = 0

        else:
            raise NotImplementedError()

        if is_qnn_params_valid:
            qnn_params = ScaleQuantParam(ConstFloat(
                scale, 32), ConstInt(zero_point, 32))
    return TfliteTensor(tensor_idx, tensor, buffer, qnn_params)


class TfliteOpConvertor(object):
    """TfliteOpConvertor

    The convertor is modified from tvm.relay.frontend.tflite.py
    """

    def __init__(self, model, subgraph) -> None:
        self.model = model
        self.subgraph = subgraph

    def get_output_tensors(self, op):
        outputs = op.OutputsAsNumpy()
        return self.get_tensors(outputs)

    def get_input_tensors(self, op):
        inputs = op.InputsAsNumpy()
        return self.get_tensors(inputs)

    def get_tensors(self, tensors_idx_list):
        ret = []
        for idx in tensors_idx_list:
            if idx < 0:
                ret.append(TfliteTensor(idx, 0, 0))
                continue

            ret.append(get_tflite_tensor(self.model, self.subgraph, idx))
        return ret

    def convert(self, op: Operator, ctx: "ConvertContext"):
        assert isinstance(op, Operator)

        ret = self._convert(op, ctx)

        if ret is not None:
            for k, v in ret.items():
                tensor_idx = v.tensor_idx
                ctx.add_tensor(get_tflite_tensor_name(
                    self.subgraph, tensor_idx), v)

    def _convert(self, op, ctx):
        raise NotImplementedError("Override _convert method is required.")

    def _convert_elementwise(self, op, ctx, op_name):
        # TODO: add parser logic for fused_activation_fn
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2

        lhs_tensor = input_tensors[0]
        rhs_tensor = input_tensors[1]
        lhs_tensor_name = get_tflite_tensor_name(
            self.subgraph, lhs_tensor.tensor_idx)
        rhs_tensor_name = get_tflite_tensor_name(
            self.subgraph, rhs_tensor.tensor_idx)
        lhs_tensor_ir = ctx.get_tensor(lhs_tensor_name)
        rhs_tensor_ir = ctx.get_tensor(rhs_tensor_name)
        assert lhs_tensor_ir.layout == rhs_tensor_ir.layout, (
            f"Layout mismatch: {lhs_tensor_ir.layout} vs. {rhs_tensor_ir.layout}")

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1
        output_tensor = output_tensors[0]
        output_tensor_name = get_tflite_tensor_name(
            self.subgraph, output_tensor.tensor_idx)
        output_tensor_ir = Tensor(
            output_tensor.shape, output_tensor.dtype,
            layout=lhs_tensor_ir.layout,
            name=output_tensor_name,
            quant_params=output_tensor.qnn_params,
            tensor_idx=output_tensor.tensor_idx
        )

        inputs_dict = {
            "lhs": lhs_tensor_ir,
            "rhs": rhs_tensor_ir
        }

        outputs_dict = {
            "output": output_tensor_ir
        }

        elem_op = Op.NamedOp(
            op_name,
            inputs_dict,
            outputs_dict,
            quant_params=None,
            attrs=None
        )

        return elem_op.outputs

    def _convert_pool2d(self, op, ctx, op_name):
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1
        input_tensor = input_tensors[0]
        input_tensor_name = get_tflite_tensor_name(
            self.subgraph, input_tensor.tensor_idx)
        input_tensor_ir = ctx.get_tensor(input_tensor_name)
        N, H, W, C = input_tensor.shape

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1
        output_tensor = output_tensors[0]
        output_tensor_name = get_tflite_tensor_name(
            self.subgraph, output_tensor.tensor_idx)
        output_tensor_type = output_tensor.dtype
        output_tensor_ir = Tensor(
            output_tensor.shape, output_tensor.dtype,
            layout=input_tensor_ir.layout,
            quant_params=output_tensor.qnn_params,
            tensor_idx=output_tensor.tensor_idx,
            name=output_tensor_name
        )

        assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
        op_options = op.BuiltinOptions()
        pool2d_options = Pool2DOptions()
        pool2d_options.Init(op_options.Bytes, op_options.Pos)
        stride_h = pool2d_options.StrideH()
        stride_w = pool2d_options.StrideW()
        padding = pool2d_options.Padding()
        filter_h = pool2d_options.FilterHeight()
        filter_w = pool2d_options.FilterWidth()
        fused_activation_fn = pool2d_options.FusedActivationFunction()

        attrs = {
            "pool_size": Attribute(ExprList([ConstInt(filter_h), ConstInt(filter_w)])),
            "strides": Attribute(ExprList([ConstInt(stride_h), ConstInt(stride_w)])),
        }

        if padding == Padding.VALID:
            padding_values = ExprList([ConstInt(0), ConstInt(0)])
        elif padding == Padding.SAME:
            pad_h = filter_h // 2
            pad_w = filter_w // 2
            padding_values = ExprList([ConstInt(pad_h), ConstInt(pad_w)])
        else:
            raise RuntimeError(
                f"Padding format {padding} for operator Pool2D is not supported.")

        attrs["padding"] = Attribute(padding_values)

        output_quant_scale = output_tensor.qnn_params.scale.value
        output_quant_zero_point = output_tensor.qnn_params.zero_point.value

        if output_tensor.qnn_params is not None:
            def quantize(x): return int(
                round(x / output_quant_scale) + output_quant_zero_point)
            if fused_activation_fn == ActivationFunctionType.NONE:
                quant_params = None
            elif fused_activation_fn == ActivationFunctionType.RELU6:
                quant_params = ClipQuantParam(
                    ConstInt(
                        max(DType.make(output_tensor_type).min_limit(), quantize(0))),
                    ConstInt(
                        min(DType.make(output_tensor_type).max_limit(), quantize(6)))
                )
            elif fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
                quant_params = ClipQuantParam(
                    ConstInt(
                        max(DType.make(output_tensor_type).min_limit(), quantize(-1))),
                    ConstInt(
                        min(DType.make(output_tensor_type).max_limit(), quantize(1)))
                )
            elif fused_activation_fn == ActivationFunctionType.RELU:
                quant_params = ClipQuantParam(
                    ConstInt(
                        max(DType.make(output_tensor_type).min_limit(), quantize(0))),
                    ConstInt(DType.make(output_tensor_type).max_limit())
                )
            else:
                raise NotImplementedError(
                    f"No support for activation type {fused_activation_fn}")
        else:
            quant_params = None
            if fused_activation_fn == ActivationFunctionType.NONE:
                pass
            elif fused_activation_fn == ActivationFunctionType.RELU6:
                attrs["activation"] = ActivationAttr("relu6")
            elif fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
                attrs["activation"] = ActivationAttr("relu_n1_to_1")
            elif fused_activation_fn == ActivationFunctionType.RELU:
                attrs["activation"] = ActivationAttr("relu")
            else:
                raise NotImplementedError(
                    f"No support for activation type {fused_activation_fn}")

        pool_op = Op.NamedOp(
            op_name,
            {
                "inputs": input_tensor_ir
            },
            {
                "outputs": output_tensor_ir
            },
            quant_params=quant_params,
            attrs=attrs
        )

        return pool_op.outputs


class Conv2dConvertor(TfliteOpConvertor):
    def _convert(self, op, ctx):
        assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
        input_tensors = self.get_input_tensors(op)
        assert len(
            input_tensors) >= 2, "Conv2d should have at least two input tensors"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        input_tensor_ir = ctx.get_tensor(
            get_tflite_tensor_name(self.subgraph, input_tensor_idx))
        # TFlite layout is NHWC
        if input_tensor_ir.layout is None:
            input_tensor_ir.layout = "NHWC"
        else:
            assert input_tensor_ir.layout == "NHWC", (
                f"Expect TFlite Conv2d has input layout: NHWC, but get {input_tensor_ir.layout}")
        weight_tensor = input_tensors[1]
        weight_tensor_name = get_tflite_tensor_name(
            self.subgraph, weight_tensor.tensor_idx)
        if ctx.has_tensor(weight_tensor_name):
            weight_tensor_ir = ctx.get_tensor(weight_tensor_name)

            if weight_tensor_ir.layout != "RSKC":
                raise NotImplementedError(
                    "No support for other layout of weight except for 'RSKC'")
        else:
            weight_tensor_type = _decode_type(weight_tensor.tensor.Type())
            # TFlite Conv2d weight layout: [K, R, S, C]
            # We need [R, S, K, C]
            weight_value = weight_tensor.value.transpose((1, 2, 0, 3))
            weight_tensor_ir = ConstTensor(
                weight_tensor.shape, weight_tensor.dtype, weight_value,
                layout="RSKC", name=weight_tensor_name, tensor_idx=weight_tensor.tensor_idx)

        if len(input_tensors) > 2:
            assert len(input_tensors) == 3
            bias_tensor = input_tensors[2]
            bias_tensor_name = get_tflite_tensor_name(
                self.subgraph, bias_tensor.tensor_idx)
            if ctx.has_tensor(bias_tensor_name):
                bias_tensor_ir = ctx.get_tensor(bias_tensor_name)

                if bias_tensor_ir.layout != "K":
                    raise NotImplementedError(
                        "No support for other layout of bias except for 'K'")
            else:
                bias_tensor_type = _decode_type(bias_tensor.tensor.Type())
                bias_tensor_ir = ConstTensor(
                    bias_tensor.shape, bias_tensor.dtype, bias_tensor.value,
                    layout="K", name=bias_tensor_name, tensor_idx=bias_tensor.tensor_idx
                )
        else:
            bias_tensor_ir = None

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "Conv2d should have one output tensor"
        output_tensor = output_tensors[0]
        output_tensor_name = get_tflite_tensor_name(
            self.subgraph, output_tensor.tensor_idx)
        output_tensor_type = _decode_type(output_tensor.tensor.Type())
        output_tensor_ir = Tensor(
            output_tensor.shape, output_tensor.dtype, layout="NHWC",
            name=output_tensor_name, quant_params=output_tensor.qnn_params,
            tensor_idx=output_tensor.tensor_idx
        )

        op_options = op.BuiltinOptions()
        conv_options = Conv2DOptions()
        conv_options.Init(op_options.Bytes, op_options.Pos)

        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()
        padding = conv_options.Padding()
        fused_activation_fn = conv_options.FusedActivationFunction()

        N, H, W, C = input_tensor.shape
        K, R, S, _C = weight_tensor.shape
        assert C == _C, "Input channels in input and weight mismatch"

        if padding == Padding.VALID:
            padding_values = ExprList([ConstInt(0), ConstInt(0)])
        elif padding == Padding.SAME:
            pad_h = ((R - 1) * dilation_h + 1) // 2
            pad_w = ((S - 1) * dilation_w + 1) // 2
            padding_values = ExprList([ConstInt(pad_h), ConstInt(pad_w)])
        else:
            raise RuntimeError(
                f"Padding format {padding} is not supported for operator Conv."
            )

        input_ir_dict = {
            "inputs": input_tensor_ir,
            "weight": weight_tensor_ir,
        }
        if bias_tensor_ir is not None:
            input_ir_dict["bias"] = bias_tensor_ir

        output_ir_dict = {
            "output": output_tensor_ir
        }

        attrs = {
            "strides": Attribute(ExprList([ConstInt(stride_h), ConstInt(stride_w)])),
            "dilation": Attribute(ExprList([ConstInt(dilation_h), ConstInt(dilation_w)])),
            "padding": Attribute(padding_values),
            "use_bias": Attribute(ConstInt(int(bias_tensor_ir is not None)))
        }

        output_quant_scale = output_tensor.qnn_params.scale.value
        output_quant_zero_point = output_tensor.qnn_params.zero_point.value

        if output_tensor.qnn_params is not None:
            def quantize(x): return int(
                round(x / output_quant_scale) + output_quant_zero_point)
            if fused_activation_fn == ActivationFunctionType.NONE:
                quant_params = None
            elif fused_activation_fn == ActivationFunctionType.RELU6:
                quant_params = ClipQuantParam(
                    ConstInt(
                        max(DType.make(output_tensor_type).min_limit(), quantize(0))),
                    ConstInt(
                        min(DType.make(output_tensor_type).max_limit(), quantize(6)))
                )
            elif fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
                quant_params = ClipQuantParam(
                    ConstInt(
                        max(DType.make(output_tensor_type).min_limit(), quantize(-1))),
                    ConstInt(
                        min(DType.make(output_tensor_type).max_limit(), quantize(1)))
                )
            elif fused_activation_fn == ActivationFunctionType.RELU:
                quant_params = ClipQuantParam(
                    ConstInt(
                        max(DType.make(output_tensor_type).min_limit(), quantize(0))),
                    ConstInt(DType.make(output_tensor_type).max_limit())
                )
            else:
                raise NotImplementedError(
                    f"No support for activation type {fused_activation_fn}")
        else:
            quant_params = None
            if fused_activation_fn == ActivationFunctionType.NONE:
                pass
            elif fused_activation_fn == ActivationFunctionType.RELU6:
                attrs["activation"] = ActivationAttr("relu6")
            elif fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
                attrs["activation"] = ActivationAttr("relu_n1_to_1")
            elif fused_activation_fn == ActivationFunctionType.RELU:
                attrs["activation"] = ActivationAttr("relu")
            else:
                raise NotImplementedError(
                    f"No support for activation type {fused_activation_fn}")

        conv_op = Op.NamedOp(
            Op.OpName.ConvOp.Conv2d,
            input_ir_dict,
            output_ir_dict,
            quant_params,
            attrs
        )

        return conv_op.outputs


class AddConvertor(TfliteOpConvertor):
    def _convert(self, op, ctx):
        return self._convert_elementwise(op, ctx, Op.OpName.ElementwiseOp.Add)


class AveragePool2dConvertor(TfliteOpConvertor):
    def _convert(self, op, ctx):
        return self._convert_pool2d(op, ctx, Op.OpName.PoolingOp.AveragePool2d)


class DepthwiseConv2dConvertor(TfliteOpConvertor):
    def _convert(self, op, ctx):
        assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
        input_tensors = self.get_input_tensors(op)
        assert len(
            input_tensors) >= 2, "DepthwiseConv2d should have at least two input tensors"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        input_tensor_ir = ctx.get_tensor(
            get_tflite_tensor_name(self.subgraph, input_tensor_idx))
        # TFlite layout is NHWC
        if input_tensor_ir.layout is None:
            input_tensor_ir.layout = "NHWC"
        else:
            assert input_tensor_ir.layout == "NHWC", (
                f"Expect TFlite Conv2d has input layout: NHWC, but get {input_tensor_ir.layout}")
        weight_tensor = input_tensors[1]
        weight_tensor_name = get_tflite_tensor_name(
            self.subgraph, weight_tensor.tensor_idx)

        if len(input_tensors) > 2:
            assert len(input_tensors) == 3
            bias_tensor = input_tensors[2]
            bias_tensor_name = get_tflite_tensor_name(
                self.subgraph, bias_tensor.tensor_idx)
            if ctx.has_tensor(bias_tensor_name):
                bias_tensor_ir = ctx.get_tensor(bias_tensor_name)

                if bias_tensor_ir.layout != "K":
                    raise NotImplementedError(
                        "No support for other layout of bias except for 'K'")
            else:
                bias_tensor_type = _decode_type(bias_tensor.tensor.Type())
                bias_tensor_ir = ConstTensor(
                    bias_tensor.shape, bias_tensor.dtype, bias_tensor.value,
                    layout="K", name=bias_tensor_name, tensor_idx=bias_tensor.tensor_idx
                )
        else:
            bias_tensor_ir = None

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "Conv2d should have one output tensor"
        output_tensor = output_tensors[0]
        output_tensor_name = get_tflite_tensor_name(
            self.subgraph, output_tensor.tensor_idx)
        output_tensor_type = _decode_type(output_tensor.tensor.Type())
        output_tensor_ir = Tensor(
            output_tensor.shape, output_tensor.dtype, layout="NHWC",
            name=output_tensor_name, quant_params=output_tensor.qnn_params,
            tensor_idx=output_tensor.tensor_idx
        )

        op_options = op.BuiltinOptions()
        conv_options = DepthwiseConv2DOptions()
        conv_options.Init(op_options.Bytes, op_options.Pos)

        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()
        padding = conv_options.Padding()
        fused_activation_fn = conv_options.FusedActivationFunction()
        depth_multiplier = conv_options.DepthMultiplier()

        N, H, W, C = input_tensor.shape
        one, R, S, _C = weight_tensor.shape
        assert one == 1
        assert C * depth_multiplier == _C, "Input channels in input and weight mismatch"

        if ctx.has_tensor(weight_tensor_name):
            weight_tensor_ir = ctx.get_tensor(weight_tensor_name)

            if weight_tensor_ir.layout != "KRSC":
                raise NotImplementedError(
                    "No support for other layout of weight except for 'KRSC'")
        else:
            weight_tensor_type = _decode_type(weight_tensor.tensor.Type())
            # TFlite Depthwise Conv2d weight layout: [1, R, S, K*mult]
            # We need [mult, R, S, K]
            weight_value = weight_tensor.value.reshape(
                [R, S, C, depth_multiplier]).transpose((3, 0, 1, 2))
            weight_tensor_ir = ConstTensor(
                weight_tensor.shape, weight_tensor.dtype, weight_value,
                layout="MRSK", name=weight_tensor_name, tensor_idx=weight_tensor.tensor_idx)

        if padding == Padding.VALID:
            padding_values = ExprList([ConstInt(0), ConstInt(0)])
        elif padding == Padding.SAME:
            pad_h = ((R - 1) * dilation_h + 1) // 2
            pad_w = ((S - 1) * dilation_w + 1) // 2
            padding_values = ExprList([ConstInt(pad_h), ConstInt(pad_w)])
        else:
            raise RuntimeError(
                f"Padding format {padding} is not supported for operator Conv."
            )

        input_ir_dict = {
            "inputs": input_tensor_ir,
            "weight": weight_tensor_ir,
        }
        if bias_tensor_ir is not None:
            input_ir_dict["bias"] = bias_tensor_ir

        output_ir_dict = {
            "output": output_tensor_ir
        }

        attrs = {
            "strides": Attribute(ExprList([ConstInt(stride_h), ConstInt(stride_w)])),
            "dilation": Attribute(ExprList([ConstInt(dilation_h), ConstInt(dilation_w)])),
            "padding": Attribute(padding_values),
            "use_bias": Attribute(ConstInt(int(bias_tensor_ir is not None)))
        }

        output_quant_scale = output_tensor.qnn_params.scale.value
        output_quant_zero_point = output_tensor.qnn_params.zero_point.value

        if output_tensor.qnn_params is not None:
            def quantize(x): return int(
                round(x / output_quant_scale) + output_quant_zero_point)
            if fused_activation_fn == ActivationFunctionType.NONE:
                quant_params = None
            elif fused_activation_fn == ActivationFunctionType.RELU6:
                quant_params = ClipQuantParam(
                    ConstInt(
                        max(DType.make(output_tensor_type).min_limit(), quantize(0))),
                    ConstInt(
                        min(DType.make(output_tensor_type).max_limit(), quantize(6)))
                )
            elif fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
                quant_params = ClipQuantParam(
                    ConstInt(
                        max(DType.make(output_tensor_type).min_limit(), quantize(-1))),
                    ConstInt(
                        min(DType.make(output_tensor_type).max_limit(), quantize(1)))
                )
            elif fused_activation_fn == ActivationFunctionType.RELU:
                quant_params = ClipQuantParam(
                    ConstInt(
                        max(DType.make(output_tensor_type).min_limit(), quantize(0))),
                    ConstInt(DType.make(output_tensor_type).max_limit())
                )
            else:
                raise NotImplementedError(
                    f"No support for activation type {fused_activation_fn}")
        else:
            quant_params = None
            if fused_activation_fn == ActivationFunctionType.NONE:
                pass
            elif fused_activation_fn == ActivationFunctionType.RELU6:
                attrs["activation"] = ActivationAttr("relu6")
            elif fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
                attrs["activation"] = ActivationAttr("relu_n1_to_1")
            elif fused_activation_fn == ActivationFunctionType.RELU:
                attrs["activation"] = ActivationAttr("relu")
            else:
                raise NotImplementedError(
                    f"No support for activation type {fused_activation_fn}")

        conv_op = Op.NamedOp(
            Op.OpName.ConvOp.DepthwiseConv2d,
            input_ir_dict,
            output_ir_dict,
            quant_params,
            attrs
        )

        return conv_op.outputs


class PadConvertor(TfliteOpConvertor):
    def _convert(self, op, ctx):
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) in [2, 3]

        if len(input_tensors) == 3:
            assert input_tensors[0].dtype == input_tensors[2].dtype

        input_tensor = input_tensors[0]
        input_tensor_name = get_tflite_tensor_name(
            self.subgraph, input_tensor.tensor_idx)
        input_tensor_ir = ctx.get_tensor(input_tensor_name)

        paddings = input_tensors[1].value
        paddings = ExprList([ExprList([ConstInt(int(x))
                            for x in y]) for y in paddings])

        if input_tensor.qnn_params is not None:
            pad_value = input_tensor.qnn_params.zero_point
        else:
            pad_value = make_const(0, input_tensor.dtype)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1
        output_tensor = output_tensors[0]
        output_tensor_name = get_tflite_tensor_name(
            self.subgraph, output_tensor.tensor_idx
        )
        output_tensor_ir = Tensor(
            output_tensor.shape, output_tensor.dtype,
            layout=input_tensor_ir.layout,
            name=output_tensor_name,
            quant_params=output_tensor.qnn_params,
            tensor_idx=output_tensor.tensor_idx
        )

        pad_op = Op.NamedOp(
            Op.OpName.PadOp.Pad,
            {
                "input": input_tensor_ir
            },
            {
                "output": output_tensor_ir
            },
            quant_params=None,
            attrs={
                "padding": Attribute(paddings),
                "pad_value": Attribute(pad_value)
            }
        )

        return pad_op.outputs


class ResizeNearestNeighborConvertor(TfliteOpConvertor):
    def _convert(self, op, ctx):
        raise NotImplementedError()


class MaxPool2dConvertor(TfliteOpConvertor):
    def _convert(self, op, ctx):
        return self._convert_pool2d(op, ctx, Op.OpName.PoolingOp.MaxPool2d)


class MeanConvertor(TfliteOpConvertor):
    def _convert(self, op, ctx):
        raise NotImplementedError()


class TransposeConvertor(TfliteOpConvertor):
    def _convert(self, op, ctx):
        raise NotImplementedError()


class FullyConnectedConvertor(TfliteOpConvertor):
    def _convert(self, op, ctx):
        raise NotImplementedError()


class ReshapeConvertor(TfliteOpConvertor):
    def _convert(self, op, ctx):
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) in [1, 2]
        input_tensor = input_tensors[0]
        input_tensor_name = get_tflite_tensor_name(
            self.subgraph, input_tensor.tensor_idx)
        input_tensor_ir = ctx.get_tensor(input_tensor_name)

        if len(input_tensors) == 2:
            shape_tensor = input_tensors[1]
            target_shape = [ConstInt(int(x)) for x in shape_tensor.value]
        else:
            assert op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
            op_options = op.BuiltinOptions()
            reshape_options = ReshapeOptions()
            reshape_options.Init(op_options.Bytes, op_options.Pos)
            target_shape = [ConstInt(int(x))
                            for x in (reshape_options.NewShapeAsNumpy())]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1
        output_tensor = output_tensors[0]
        output_tensor_name = get_tflite_tensor_name(
            self.subgraph, output_tensor.tensor_idx)
        output_tensor_ir = Tensor(
            output_tensor.shape, output_tensor.dtype,
            layout=None,  # TODO: how to determine the layout?
            name=output_tensor_name,
            quant_params=output_tensor.qnn_params,
            tensor_idx=output_tensor.tensor_idx
        )

        attrs = {
            "target_shape": Attribute(ExprList(target_shape))
        }

        reshape_op = Op.NamedOp(
            Op.OpName.ScalingOp.Reshape,
            {
                "inputs": input_tensor_ir
            },
            {
                "outputs": output_tensor_ir
            },
            quant_params=None,
            attrs=attrs
        )

        return reshape_op.outputs


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
    "RESHAPE": ReshapeConvertor
}


class ConvertContext(object):
    def __init__(self) -> None:
        self.tensor_ctx: Dict[str, Tensor] = {}
        self.shape_dict = {}
        self.dtype_dict = {}
        self.layout_dict = {}
        self.model_input_names = []
        self.model_output_names = []

    def has_tensor(self, name: str):
        return name in self.tensor_ctx

    def get_tensor(self, name: str):
        return self.tensor_ctx[name]

    def add_tensor_shape(
        self,
        name: str,
        shape: ShapeType,
    ):
        self.shape_dict[name] = shape

    def update_tensor(self, name: str, tensor):
        self.add_tensor(name, tensor, override=True)

    def add_tensor_dtype(
        self,
        name: str,
        dtype: GeneralDType
    ):
        self.dtype_dict[name] = dtype

    def add_model_input_name(self, name):
        self.model_input_names.append(name)

    def add_model_output_name(self, name):
        self.model_output_names.append(name)

    def add_input_tensor(
            self,
            name: str,
            shape: ShapeType,
            dtype: GeneralDType,
            tensor_idx: int,
            layout: Optional[str] = None,
            quant_params: Optional[TensorQuantParam] = None,
            override: bool = False) -> None:
        if name in self.tensor_ctx:
            if not override:
                raise RuntimeError(
                    f"Try to override an existing tensor {name}")
        else:
            self.tensor_ctx[name] = Tensor(
                shape, dtype, produce_op=None, out_idx="", layout=layout, quant_params=quant_params, name=name,
                tensor_idx=tensor_idx)

    def add_tensor(self, name: str, tensor: Tensor, override=False) -> None:
        if name in self.tensor_ctx:
            if not override:
                raise RuntimeError(
                    f"Try to override an existing tensor {name}")
        else:
            self.tensor_ctx[name] = tensor

    def make_graph(self):
        inputs_dict = {k: self.get_tensor(k) for k in self.model_input_names}
        outputs_dict = {k: self.get_tensor(k) for k in self.model_output_names}
        subgraph = SubGraph(inputs_dict, outputs_dict)
        return Graph({"subgraph0": subgraph}, inputs_dict, outputs_dict)


class TfliteConvertor(object):
    def __init__(self, path: str) -> None:
        self.model_path = path
        print(f"Using tflite version {tflite.__version__}.")

    def parse(self):
        model = load_tflite_model(self.model_path)
        assert model.SubgraphsLength() == 1, "only support one subgraph"
        subgraph = model.Subgraphs(0)
        num_ops = subgraph.OperatorsLength()
        op_map = get_tflite_builtin_op_map(tflite.BuiltinOperator())
        ctx = ConvertContext()

        model_inputs = subgraph.InputsAsNumpy()
        model_outputs = subgraph.OutputsAsNumpy()

        for input_idx in model_inputs:
            tensor = get_tflite_tensor(model, subgraph, input_idx)
            input_name = get_tflite_tensor_name(subgraph, input_idx)
            ctx.add_input_tensor(input_name, tensor.shape, tensor.dtype,
                                 layout=None, quant_params=tensor.qnn_params,
                                 tensor_idx=input_idx)
            ctx.add_model_input_name(input_name)

        for output_idx in model_outputs:
            output_name = get_tflite_tensor_name(subgraph, output_idx)
            ctx.add_model_output_name(output_name)

        for i in range(num_ops):
            op = subgraph.Operators(i)
            self.parse_op(model, subgraph, op_map, op, ctx)

        return ctx.make_graph()

    def parse_op(self, model, subgraph, op_map, op, ctx):
        op_code = op.OpcodeIndex()
        op_code = model.OperatorCodes(op_code)
        try:
            op_code_id = max(op_code.DeprecatedBuiltinCode(),
                             op_code.BuiltinCode())
        except AttributeError:
            op_code_id = op_code.BuiltinCode()
        op_name = op_map[op_code_id]

        if op_name not in CONVERT_MAP:
            raise RuntimeError(f"Op {op_name} is not supported yet.")
        convertor = CONVERT_MAP[op_name]
        convertor(model, subgraph).convert(op, ctx)


if __name__ == "__main__":
    path = "mcunet-5fps_vww.tflite"
    convertor = TfliteConvertor(path)
    graph = convertor.parse()
    print(graph)
