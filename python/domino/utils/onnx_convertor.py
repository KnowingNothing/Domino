from math import prod
import onnx
import numpy as np
from onnx.numpy_helper import to_array
from domino.graph_ir import ConstTensor, Tensor, Attribute, Op, SubGraph, Graph
from domino.program_ir import ConstInt, ConstUInt, ConstFloat, ConstString, ExprList
from domino.type_system import DType


def get_type(elem_type):
    """Converts onnx integer datatype to numpy datatype"""
    # If a string was passed instead of a tensor type, it does not need
    # conversion and can be returned.
    if isinstance(elem_type, str):
        return elem_type

    try:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    except ImportError as e:
        raise ImportError(
            "Unable to import onnx which is required {}".format(e))

    try:
        from onnx import TensorProto
    except ImportError as e:
        raise ImportError(
            "Unable to import TensorProto from onnx {}".format(e))

    # Onnx mapping converts bfloat16 to float16 because
    # numpy does not have a bfloat16 data type. However,
    # tvm has one, so we force the return type to be bfloat16
    if elem_type == int(TensorProto.BFLOAT16):
        return "bfloat16"
    return str(TENSOR_TYPE_TO_NP_TYPE[elem_type])


class ONNXOpConvertor(object):
    def __init__(self, opset) -> None:
        self.opset = opset

    def convert(self, ctx, op):
        attrs = ctx._parse_attr(op.attribute)
        inputs = [ctx.get_tensor(name) for name in op.input if name != ""]
        op_name = op.op_type

        versions = sorted([int(x.replace("convert_v", ""))
                          for x in dir(self) if x.startswith("convert_v")])
        assert len(versions) > 0
        version = versions[0]
        for v in versions:
            if v <= self.opset:
                version = v
            else:
                break
        convertor_name = f"convert_v{version}"

        if not hasattr(self, convertor_name):
            raise RuntimeError(
                f"No implementation for {convertor_name} in class {self.__class__}")
        convertor = getattr(self, convertor_name)
        outputs = convertor(ctx, op, inputs, attrs)

        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print(op)
        if outputs is not None:
            for name, out in outputs.items():
                ctx.add_tensor(out.tensor_idx, out)
        #         print(out.shape)
        # print()


class ConvConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        data = inputs[0]  # NCHW layout
        if data.layout is None:
            data.layout = "NCHW"
        else:
            assert data.layout == "NCHW"

        filter = inputs[1]  # KCRS layout
        if filter.layout is None:
            filter.layout = "KCRS"
        else:
            assert filter.layout == "KCRS"

        N, C, H, W = data.shape
        K, _C, R, S = filter.shape

        op_attrs = {}

        if "auto_pad" in attrs:
            raise NotImplementedError("Auto_pad is not supported currently.")

        if len(attrs["pads"]) == 4:
            # assert attrs["pads"][0] == attrs["pads"][1]
            # assert attrs["pads"][2] == attrs["pads"][3]
            # In ONNX, pad is [dim_0_beg, dim_1_beg, ..., dim_0_end, dim_1_end, ...]
            # We want [dim_0_beg, dim_0_end, dim_1_beg, dim_1_end, ...]
            pad_values = ExprList(
                [ConstInt(attrs["pads"][0]), ConstInt(attrs["pads"][2]), ConstInt(attrs["pads"][1]), ConstInt(attrs["pads"][3])])
            op_attrs["padding"] = Attribute(pad_values)
        elif len(attrs["pads"]) == 2:
            pad_values = ExprList(
                [ConstInt(attrs["pads"][0]), ConstInt(attrs["pads"][0]), ConstInt(attrs["pads"][1]), ConstInt(attrs["pads"][1])])
            op_attrs["padding"] = Attribute(pad_values)
        else:
            raise ValueError(
                f"Can't understand the padding attribute: {attrs['pads']}")

        if "strides" in attrs:
            if len(attrs["strides"]) == 2:
                strides = ExprList(
                    [ConstInt(attrs["strides"][0]), ConstInt(attrs["strides"][1])])
                op_attrs["strides"] = Attribute(strides)
            else:
                raise ValueError(
                    f"Can't understand the strides attribute: {attrs['strides']}")
        else:
            strides = ExprList([ConstInt(1), ConstInt(1)])

        if "dilations" in attrs:
            if len(attrs["dilations"]) == 2:
                dilations = ExprList(
                    [ConstInt(attrs["dilations"][0]), ConstInt(attrs["dilations"][1])])
                op_attrs["dilation"] = Attribute(dilations)
            else:
                raise ValueError(
                    f"Can't understand the dilations attribute: {attrs['dilations']}")
        else:
            dilations = ExprList([ConstInt(1), ConstInt(1)])

        op_attrs["use_bias"] = Attribute(ConstUInt((len(inputs) == 3)))

        if "group" not in attrs:
            groups = 1
        else:
            groups = attrs["group"]

        inputs_dict = {
            "inputs": data,
            "weight": filter,
        }

        if op_attrs["use_bias"]:
            inputs_dict["bias"] = inputs[2]

        kh = (R - 1) * dilations[0].value + 1
        kw = (S - 1) * dilations[1].value + 1
        P = (H + pad_values[0].value +
             pad_values[1].value - kh) // attrs["strides"][0] + 1
        Q = (W + pad_values[2].value +
             pad_values[3].value - kw) // attrs["strides"][1] + 1

        name = op.output[0]
        shape = [N, K, P, Q]

        output = Tensor(
            shape,
            data.dtype,  # TODO: how to determine the data type of output?
            layout="NCHW",
            name=name,
            tensor_idx=name
        )

        outputs_dict = {
            "output": output
        }

        if groups > 1:
            if groups == C:
                # depthwise conv
                filter.layout = "KMRS"
                assert _C == 1, f"Expect Depthwise Conv has in_channel = 1, but get {_C}"

                conv_op = Op.NamedOp(
                    Op.OpName.ConvOp.DepthwiseConv2d,
                    inputs_dict,
                    outputs_dict,
                    attrs=op_attrs
                )
            else:
                raise NotImplementedError("Grouped Conv is not supported yet.")
        else:
            assert C == _C
            conv_op = Op.NamedOp(
                Op.OpName.ConvOp.Conv2d,
                inputs_dict,
                outputs_dict,
                attrs=op_attrs
            )

        return conv_op.outputs


class ReluConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        data = inputs[0]
        assert len(inputs) == 1

        output_names = [x for x in op.output]
        assert len(output_names) == 1

        output = Tensor(
            data.shape,
            data.dtype,
            layout=data.layout,
            name=output_names[0],
            tensor_idx=output_names[0]
        )

        inputs_dict = {
            "inputs": data
        }

        outputs_dict = {
            "output": output
        }

        relu_op = Op.NamedOp(
            Op.OpName.ActivationOp.ReLU,
            inputs_dict,
            outputs_dict
        )

        return relu_op.outputs


class PoolConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs, op_name):
        data = inputs[0]
        assert len(inputs) == 1
        if data.layout is None:
            data.layout = "NCHW"
        else:
            assert data.layout == "NCHW", "Only NCHW layout 2D pooling is supported."

        N, C, H, W = data.shape

        op_attrs = {}

        if "auto_pad" in attrs:
            raise NotImplementedError("Auto_pad is not supported currently.")

        if len(attrs["pads"]) == 4:
            # assert attrs["pads"][0] == attrs["pads"][1]
            # assert attrs["pads"][2] == attrs["pads"][3]
            # In ONNX, pad is [dim_0_beg, dim_1_beg, ..., dim_0_end, dim_1_end, ...]
            # We want [dim_0_beg, dim_0_end, dim_1_beg, dim_1_end, ...]
            pad_values = ExprList(
                [ConstInt(attrs["pads"][0]), ConstInt(attrs["pads"][2]), ConstInt(attrs["pads"][1]), ConstInt(attrs["pads"][3])])
            op_attrs["padding"] = Attribute(pad_values)
        elif len(attrs["pads"]) == 2:
            pad_values = ExprList(
                [ConstInt(attrs["pads"][0]), ConstInt(attrs["pads"][0]), ConstInt(attrs["pads"][1]), ConstInt(attrs["pads"][1])])
            op_attrs["padding"] = Attribute(pad_values)
        else:
            raise ValueError(
                f"Can't understand the padding attribute: {attrs['pads']}")

        if len(attrs["strides"]) == 2:
            strides = ExprList(
                [ConstInt(attrs["strides"][0]), ConstInt(attrs["strides"][1])])
            op_attrs["strides"] = Attribute(strides)
        else:
            raise ValueError(
                f"Can't understand the strides attribute: {attrs['strides']}")

        if len(attrs["kernel_shape"]) == 2:
            pool_size = ExprList(
                [ConstInt(attrs["kernel_shape"][0]), ConstInt(attrs["kernel_shape"][1])])
            op_attrs["pool_size"] = Attribute(pool_size)
        else:
            raise ValueError(
                f"Only support 2D pooling currently.")

        kh = attrs["kernel_shape"][0]
        kw = attrs["kernel_shape"][1]
        P = (H + pad_values[0].value +
             pad_values[1].value - kh) // attrs["strides"][0] + 1
        Q = (W + pad_values[2].value +
             pad_values[3].value - kw) // attrs["strides"][1] + 1

        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            [N, C, P, Q],
            data.dtype,
            layout=data.layout,
            name=output_name,
            tensor_idx=output_name
        )

        # TODO: support other versions of pooling
        if op_name == Op.OpName.PoolingOp.MaxPool:
            op_name = Op.OpName.PoolingOp.MaxPool2d
        elif op_name == Op.OpName.PoolingOp.AveragePool:
            op_name = Op.OpName.PoolingOp.AveragePool2d
        else:
            raise NotImplementedError(
                f"Pool operator {op_name} is not supported yet.")

        pool_op = Op.NamedOp(
            op_name,
            {
                "inputs": data
            },
            {
                "output": output
            },
            attrs=op_attrs
        )

        return pool_op.outputs


class MaxPoolConvertor(PoolConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        return super().convert_v1(ctx, op, inputs, attrs, Op.OpName.PoolingOp.MaxPool)


class AveragePoolConvertor(PoolConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        return super().convert_v1(ctx, op, inputs, attrs, Op.OpName.PoolingOp.AveragePool)


class ElementwiseConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs, op_name):
        assert len(inputs) == 2
        lhs = inputs[0]
        rhs = inputs[1]
        # TODO: how to support broadcast?
        assert lhs.dtype == rhs.dtype
        if len(rhs.shape) > 0:
            # assert lhs.shape == rhs.shape, (
            #     f"\n{op}: {op_name} operand shapes mismatch: {lhs.shape} vs {rhs.shape}")
            if lhs.shape == rhs.shape:
                if lhs.layout is None:
                    lhs.layout = rhs.layout
                elif rhs.layout is None:
                    rhs.layout = lhs.layout
                else:
                    assert lhs.layout == rhs.layout
                output_shape = lhs.shape
                output_layout = lhs.layout
            else:
                op_name = Op.OpName.elementwise_to_broadcast_op(op_name)
                output_shape = []
                output_layout = None
                ldim = len(lhs.shape)
                rdim = len(rhs.shape)
                i = ldim - 1
                j = rdim - 1
                while i >= 0 and j >= 0:
                    if lhs.shape[i] == rhs.shape[j]:
                        output_shape.append(lhs.shape[i])
                    elif lhs.shape[i] == 1:
                        output_shape.append(rhs.shape[j])
                    elif rhs.shape[j] == 1:
                        output_shape.append(lhs.shape[i])
                    else:
                        raise ValueError(
                            f"Can't broadcast the shape for {op_name}: {lhs.shape} vs {rhs.shape}")
                    i -= 1
                    j -= 1
                while i >= 0:
                    output_shape.append(lhs.shape[i])
                    i -= 1
                while j >= 0:
                    output_shape.append(rhs.shape[j])
                    j -= 1
                output_shape = list(reversed(output_shape))
        else:
            # rhs.shape = 0, scalar
            output_shape = lhs.shape
            output_layout = lhs.layout

            op_name = Op.OpName.elementwise_to_tensor_scalar_op(op_name)

        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            output_shape,
            lhs.dtype,
            layout=output_layout,
            name=output_name,
            tensor_idx=output_name
        )

        elem_op = Op.NamedOp(
            op_name,
            {
                "lhs": lhs,
                "rhs": rhs
            },
            {
                "output": output
            }
        )

        return elem_op.outputs


class AddConvertor(ElementwiseConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        return super().convert_v1(ctx, op, inputs, attrs, Op.OpName.ElementwiseOp.Add)


class MulConvertor(ElementwiseConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        return super().convert_v1(ctx, op, inputs, attrs, Op.OpName.ElementwiseOp.Mul)


class PowConvertor(ElementwiseConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        return super().convert_v1(ctx, op, inputs, attrs, Op.OpName.ElementwiseOp.Pow)


class GlobalAveragePoolConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) == 1
        assert len(
            inputs[0].shape) == 4, "Only support 2D GlobalAveragePool currently."
        N, C, H, W = inputs[0].shape
        if inputs[0].layout is not None:
            assert inputs[0].layout == "NCHW"
        else:
            inputs[0].layout = "NCHW"

        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            [N, C, 1, 1],
            inputs[0].dtype,
            layout="NCHW",
            name=output_name,
            tensor_idx=output_name
        )

        global_pool_op = Op.NamedOp(
            Op.OpName.PoolingOp.GlobalAveragePool2d,
            {
                "inputs": inputs[0]
            },
            {
                "output": output
            }
        )

        return global_pool_op.outputs


class FlattenConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert "axis" in attrs
        axis = attrs["axis"]
        assert len(inputs) == 1
        data = inputs[0]

        dim0 = prod(data.shape[:axis], start=1)
        dim1 = prod(data.shape[axis:], start=1)
        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            [dim0, dim1],
            data.dtype,
            layout=None,  # TODO: how to determine the layout?
            name=output_name,
            tensor_idx=output_name
        )

        flat_op = Op.NamedOp(
            Op.OpName.ScalingOp.Flatten,
            {
                "inputs": data
            },
            {
                "output": output
            }
        )

        return flat_op.outputs


class GemmConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) in [2, 3]

        # Y = alpha * A * B + beta * C
        alpha = float(attrs.get("alpha", 1.0))
        beta = float(attrs.get("beta", 1.0))
        transA = int(attrs.get("transA", 0))
        transB = int(attrs.get("transB", 0))

        inputs_dict = {
            "inputs": inputs[0],
            "weight": inputs[1],
        }

        if len(inputs) == 3:
            inputs_dict["bias"] = inputs[2]

        op_attrs = {
            "alpha": Attribute(ConstFloat(alpha)),
            "beta": Attribute(ConstFloat(beta)),
            "transA": Attribute(ConstInt(transA)),
            "transB": Attribute(ConstInt(transB)),
            "use_bias": Attribute(ConstUInt(len(inputs) == 3))
        }

        if transA:
            K, M = inputs[0].shape
        else:
            M, K = inputs[0].shape

        if transB:
            N, _K = inputs[1].shape
        else:
            _K, N = inputs[1].shape

        assert K == _K, "Gemm channel size mismatch"

        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            [M, N],
            inputs[0].dtype,
            layout=None,  # TODO: how to determine the layout?
            name=output_name,
            tensor_idx=output_name
        )

        gemm_op = Op.NamedOp(
            Op.OpName.MatrixOp.Gemm,
            {
                "inputs": inputs[0],
                "weight": inputs[1]
            },
            {
                "output": output
            },
            attrs=op_attrs
        )

        return gemm_op.outputs


class ConstantConvertor(ONNXOpConvertor):
    def convert_v9(self, ctx, op, inputs, attrs):
        if "value" not in attrs:
            raise ValueError("Requires value attr in Constant")
        value = attrs["value"]
        if isinstance(value, bytes):
            np_value = np.asarray([0]).astype("int64")
        else:
            np_value = ctx.parse_data(value)
        dtype = str(np_value.dtype)
        output = ConstTensor(
            np_value.shape,
            dtype,
            np_value,
            layout=None,
            name=op.output[0],
            tensor_idx=op.output[0]
        )
        return {"": output}


class ClipConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) == 1
        data = inputs[0]

        minv = attrs.get("min", -float("inf"))
        maxv = attrs.get("max", float("inf"))

        op_attrs = {
            "min": Attribute(ConstFloat(minv)),
            "max": Attribute(ConstFloat(maxv))
        }

        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            data.shape,
            data.dtype,
            layout=data.layout,
            name=output_name,
            tensor_idx=output_name
        )

        clip_op = Op.NamedOp(
            Op.OpName.ActivationOp.Clip,
            {
                "inputs": data
            },
            {
                "output": output
            },
            attrs=op_attrs
        )

        return clip_op.outputs

    def convert_v11(self, ctx, op, inputs, attrs):
        op_attrs = {}
        assert 1 <= len(inputs) <= 3
        data = inputs[0]
        if len(inputs) == 3 and isinstance(inputs[2], ConstTensor):
            op_attrs["max"] = Attribute(ConstFloat(float(inputs[2].value)))
            inputs = inputs[0:2]
        if len(inputs) >= 2 and isinstance(inputs[1], ConstTensor):
            op_attrs["min"] = Attribute(ConstFloat(float(inputs[1].value)))
            inputs = inputs[0:1]
        if "min" in attrs and "max" in attrs:
            minv = attrs.get("min", -float("inf"))
            maxv = attrs.get("max", float("inf"))

            op_attrs = {
                "min": Attribute(ConstFloat(minv)),
                "max": Attribute(ConstFloat(maxv))
            }

        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            data.shape,
            data.dtype,
            layout=data.layout,
            name=output_name,
            tensor_idx=output_name
        )

        clip_op = Op.NamedOp(
            Op.OpName.ActivationOp.Clip,
            {
                "inputs": data
            },
            {
                "output": output
            },
            attrs=op_attrs
        )

        return clip_op.outputs


class SigmoidConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) == 1
        data = inputs[0]

        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            data.shape,
            data.dtype,
            layout=data.layout,
            name=output_name,
            tensor_idx=output_name
        )

        sig_op = Op.NamedOp(
            Op.OpName.ActivationOp.Sigmoid,
            {
                "inputs": data
            },
            {
                "output": output
            },
        )

        return sig_op.outputs


class ConcatConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert "axis" in attrs
        axis = attrs["axis"]

        op_attrs = {
            "axis": Attribute(ConstInt(axis))
        }

        num_inputs = len(inputs)
        assert num_inputs > 0
        output_shape = inputs[0].shape
        output_dtype = inputs[0].dtype
        output_layout = inputs[0].layout
        assert axis < len(output_shape)
        for i in range(1, num_inputs):
            cur_shape = inputs[i].shape
            assert len(cur_shape) == len(output_shape)
            output_shape[axis] += cur_shape[axis]
            assert inputs[i].dtype.type_kind == output_dtype.type_kind
            output_dtype.bits = max(output_dtype.bits, inputs[i].dtype.bits)
            assert inputs[i].layout is None or inputs[i].layout == output_layout

        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            output_shape,
            output_dtype,
            layout=output_layout,
            name=output_name,
            tensor_idx=output_name
        )

        cat_op = Op.NamedOp(
            Op.OpName.DimOrderOp.Concat,
            {
                f"inputs{i}": inputs[i] for i in range(num_inputs)
            },
            {
                "output": output
            },
            attrs=op_attrs
        )

        return cat_op.outputs


class ResizeConvertor(ONNXOpConvertor):
    def convert_v11(self, ctx, op, inputs, attrs):
        assert len(inputs) >= 2
        if len(inputs) == 3:
            data, roi, scales = inputs
        else:
            data, scales = inputs
        assert len(scales.shape) == 1
        assert scales.shape[0] == len(data.shape)

        output_shape = []
        for s, v in zip(data.shape, scales.value):
            output_shape.append(int(s * v))

        op_attrs = {}
        if "coordinate_transformation_mode" in attrs:
            op_attrs["coordinate_transformation_mode"] = Attribute(ConstString(
                str(attrs["coordinate_transformation_mode"].decode("utf-8"))))
        if "cubic_coeff_a" in attrs:
            op_attrs["cubic_coeff_a"] = Attribute(
                ConstFloat(attrs["cubic_coeff_a"]))
        if "mode" in attrs:
            op_attrs["mode"] = Attribute(ConstString(
                str(attrs["mode"].decode("utf-8"))))
        if "nearest_mode" in attrs:
            op_attrs["nearest_mode"] = Attribute(ConstString(
                str(attrs["nearest_mode"].decode("utf-8"))))

        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            output_shape,
            data.dtype,
            layout=data.layout,
            name=output_name,
            tensor_idx=output_name
        )

        if len(inputs) == 3:
            inputs_dict = {
                "inputs": data,
                "roi": roi,
                "scales": scales
            }
        else:
            inputs_dict = {
                "inputs": data,
                "scales": scales
            }
        resize_op = Op.NamedOp(
            Op.OpName.ScalingOp.Resize,
            inputs_dict,
            {
                "output": output
            },
            attrs=op_attrs
        )

        return resize_op.outputs


class ReshapeConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) == 2
        data, new_shape = inputs

        met_unknown = False
        unknown_pos = -1
        total_elems = prod(data.shape, start=1)
        remain_elems = 1

        output_shape = list(map(int, new_shape.value))
        for i in range(len(output_shape)):
            if output_shape[i] == -1:
                if met_unknown:
                    raise RuntimeError("Can't reshape with more than one '-1'")
                met_unknown = True
                unknown_pos = i
            else:
                remain_elems *= output_shape[i]

        if met_unknown:
            output_shape[unknown_pos] = total_elems // remain_elems
            assert output_shape[unknown_pos] * remain_elems == total_elems

        assert prod(data.shape, start=1) == prod(
            output_shape, start=1), (f"{op} prod({data.shape})={prod(data.shape, start=1)} vs prod({output_shape})={prod(output_shape, start=1)}")

        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            output_shape,
            data.dtype,
            layout=None,
            name=output_name,
            tensor_idx=output_name
        )
        reshape_op = Op.NamedOp(
            Op.OpName.ScalingOp.Reshape,
            {
                "inputs": data,
                "new_shape": new_shape
            },
            {
                "output": output
            },
        )

        return reshape_op.outputs


class TransposeConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) == 1
        data, = inputs

        assert "perm" in attrs
        output_shape = []
        if data.layout is not None:
            input_layout = list(data.layout)
            output_layout = [x for x in input_layout]
        else:
            output_layout = None

        for v in attrs["perm"]:
            output_shape.append(data.shape[v])
            if output_layout is not None:
                output_layout[v] = input_layout[v]

        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            output_shape,
            data.dtype,
            layout="".join(
                output_layout) if output_layout is not None else None,
            name=output_name,
            tensor_idx=output_name
        )
        trans_op = Op.NamedOp(
            Op.OpName.DimOrderOp.Transpose,
            {
                "inputs": data,
            },
            {
                "output": output
            },
            attrs={
                "perm": Attribute(
                    ExprList(
                        [ConstInt(x) for x in attrs["perm"]]
                    )
                )
            }
        )

        return trans_op.outputs


class SplitConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) == 1
        data, = inputs

        assert "axis" in attrs
        axis = attrs["axis"]
        assert axis < len(data.shape)
        assert "split" in attrs
        split = attrs["split"]
        num_outputs = len(split)

        op_attrs = {
            "axis": Attribute(ConstInt(axis)),
            "split": Attribute(ExprList([ConstInt(x) for x in split]))
        }

        output_shapes = []
        for s in split:
            tmp_shape = data.shape
            tmp_shape[axis] = s
            output_shapes.append(tmp_shape)

        output_tensors = [
            Tensor(
                output_shape,
                data.dtype,
                layout=data.layout,
                name=output_name,
                tensor_idx=output_name
            )
            for output_shape, output_name in zip(output_shapes, op.output)
        ]

        split_op = Op.NamedOp(
            Op.OpName.DimOrderOp.Split,
            {
                "inputs": data,
            },
            {
                f"output{i}": output_tensors[i] for i in range(num_outputs)
            },
            attrs=op_attrs
        )

        return split_op.outputs


class LRNConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        data = inputs[0]
        assert len(inputs) == 1
        assert data.layout is None or data.layout == "NCHW"

        output_names = [x for x in op.output]
        assert len(output_names) == 1

        axis = 1
        alpha = attrs.get("alpha", 0.0001)
        beta = attrs.get("beta", 0.75)
        bias = attrs.get("bias", 1.0)
        nsize = attrs.get("size")

        output = Tensor(
            data.shape,
            data.dtype,
            layout=data.layout,
            name=output_names[0],
            tensor_idx=output_names[0]
        )

        inputs_dict = {
            "inputs": data
        }

        outputs_dict = {
            "output": output
        }

        lrn_op = Op.NamedOp(
            Op.OpName.ActivationOp.LRN,
            inputs_dict,
            outputs_dict,
            attrs={
                "axis": Attribute(ConstInt(axis)),
                "alpha": Attribute(ConstFloat(alpha)),
                "beta": Attribute(ConstFloat(beta)),
                "bias": Attribute(ConstFloat(bias)),
                "nsize": Attribute(ConstFloat(nsize))
            }
        )

        return lrn_op.outputs


class DropoutConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) == 2
        data = inputs[0]
        ratio = inputs[1]

        output_names = [x for x in op.output]
        assert len(output_names) == 2

        output = Tensor(
            data.shape,
            data.dtype,
            layout=data.layout,
            name=output_names[0],
            tensor_idx=output_names[0]
        )

        mask = Tensor(
            data.shape,
            DType.make("int32"),
            layout=data.layout,
            name=output_names[1],
            tensor_idx=output_names[1]
        )

        inputs_dict = {
            "inputs": data,
            "ratio": ratio
        }

        outputs_dict = {
            "output": output,
            "mask": mask
        }

        drop_op = Op.NamedOp(
            Op.OpName.ActivationOp.Dropout,
            inputs_dict,
            outputs_dict,
        )

        return drop_op.outputs


class SoftmaxConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        axis = attrs.get("axis", 1)
        assert len(inputs) == 1
        data = inputs[0]
        ndim = len(data.shape)
        if axis < 0:
            axis = ndim + axis

        output_names = [x for x in op.output]
        assert len(output_names) == 1

        output = Tensor(
            data.shape,
            data.dtype,
            layout=data.layout,
            name=output_names[0],
            tensor_idx=output_names[0]
        )

        inputs_dict = {
            "inputs": data,
        }

        outputs_dict = {
            "output": output,
        }

        softmax_op = Op.NamedOp(
            Op.OpName.ReduceOp.Softmax,
            inputs_dict,
            outputs_dict,
            attrs={
                "axis": Attribute(ConstInt(axis))
            }
        )

        return softmax_op.outputs


class IdentityConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) == 1
        data = inputs[0]

        output_names = [x for x in op.output]
        assert len(output_names) == 1

        output = ConstTensor(
            data.shape,
            data.dtype,
            data.value,
            layout=data.layout,
            name=output_names[0],
            tensor_idx=output_names[0]
        )

        inputs_dict = {
            "weight": data,
        }

        outputs_dict = {
            "output": output,
        }

        id_op = Op.NamedOp(
            Op.OpName.SourceOp.Identity,
            inputs_dict,
            outputs_dict,
        )

        return id_op.outputs


class ReduceMeanConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) == 1
        data = inputs[0]
        ndim = len(data.shape)

        output_names = [x for x in op.output]
        assert len(output_names) == 1

        axes = attrs.get("axes", list(range(ndim)))
        if isinstance(axes, int):
            axes = [axes]
        axes = [x + ndim if x < 0 else x for x in axes]

        keepdims = attrs.get("keepdims", 1)
        output_shape = []
        for i, v in enumerate(data.shape):
            if i in axes:
                if keepdims:
                    output_shape.append(1)
            else:
                output_shape.append(v)

        output = Tensor(
            output_shape,
            data.dtype,
            layout=data.layout if keepdims else None,
            name=output_names[0],
            tensor_idx=output_names[0]
        )

        inputs_dict = {
            "inputs": data,
        }

        outputs_dict = {
            "output": output,
        }

        mean_op = Op.NamedOp(
            Op.OpName.ReduceOp.ReduceMean,
            inputs_dict,
            outputs_dict,
            attrs={
                "axes": Attribute(ExprList([ConstInt(x) for x in axes])),
                "keepdims": Attribute(ConstInt(keepdims))
            }
        )

        return mean_op.outputs


class ShapeConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) == 1
        data = inputs[0]

        output_names = [x for x in op.output]
        assert len(output_names) == 1

        end = attrs.get("end", len(data.shape))
        start = attrs.get("start", 0)
        output_value = data.shape[start:end]
        output_shape = [len(output_value)]

        output = ConstTensor(
            output_shape,
            "int32",
            np.array(output_value),
            layout=None,
            name=output_names[0],
            tensor_idx=output_names[0]
        )

        inputs_dict = {
            "data": data,
        }

        outputs_dict = {
            "output": output,
        }

        shape_op = Op.NamedOp(
            Op.OpName.SourceOp.Shape,
            inputs_dict,
            outputs_dict,
        )

        return shape_op.outputs


class GatherConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) == 2
        data = inputs[0]
        ndim = len(data.shape)
        indices = inputs[1]
        ind_shape = indices.shape

        output_names = [x for x in op.output]
        assert len(output_names) == 1

        axis = attrs.get("axis", 0)
        if axis < 0:
            axis = axis + ndim
        output_shape = []
        for i in range(axis):
            output_shape.append(data.shape[i])
        output_shape.extend(ind_shape)
        for j in range(axis, len(data.shape)):
            output_shape.append(data.shape[j])

        output = Tensor(
            output_shape,
            data.dtype,
            layout=None,
            name=output_names[0],
            tensor_idx=output_names[0]
        )

        inputs_dict = {
            "data": data,
        }

        outputs_dict = {
            "output": output,
        }

        gather_op = Op.NamedOp(
            Op.OpName.SparseOp.Gather,
            inputs_dict,
            outputs_dict,
            attrs={
                "axis": Attribute(ConstInt(axis))
            }
        )

        return gather_op.outputs


class MatMulConvertor(ONNXOpConvertor):
    def convert_v1(self, ctx, op, inputs, attrs):
        assert len(inputs) == 2

        inputs_dict = {
            "lhs": inputs[0],
            "rhs": inputs[1],
        }

        lhs = inputs[0]
        rhs = inputs[1]

        assert len(lhs.shape) >= 2, "Currently we only support >= 2-D Tensor"
        assert len(rhs.shape) >= 2, "Currently we only support >= 2-D Tensor"
        # assert len(lhs.shape) == len(rhs.shape), f"Currently we don't support broadcast for MatMul {lhs.shape} vs {rhs.shape}:\n{op}"
        M, K = lhs.shape[-2], lhs.shape[-1]
        K_, N = rhs.shape[-2], rhs.shape[-1]
        assert K == K_, f"MatMul K dimension mismatches: {K} vis {K_}"

        # for i in range(len(lhs.shape) - 2):
        #     assert lhs.shape[i] == rhs.shape[i], (
        #         f"Currently we don't support broadcast for MatMul. Operand shape mismatch: {lhs.shape[i]} vs {rhs.shape[i]}")
        assert lhs.dtype == rhs.dtype

        output_shape = [N, M]
        ldim = len(lhs.shape) - 2
        rdim = len(rhs.shape) - 2
        for i in range(min(ldim, rdim)):
            if lhs.shape[ldim - 1 - i] == rhs.shape[rdim - 1 - i]:
                output_shape.append(lhs.shape[ldim - 1 - i])
            elif lhs.shape[ldim - 1 - i] == 1:
                output_shape.append(rhs.shape[rdim - 1 - i])
            elif rhs.shape[rdim - 1 - i] == 1:
                output_shape.append(lhs.shape[ldim - 1 - i])
            else:
                raise ValueError(
                    f"Fail to broadcast the shape for MatMul: lhs: {lhs.shape}, rhs: {rhs.shape}")
        for i in range(ldim - min(ldim, rdim)):
            output_shape.append(lhs.shape[ldim - min(ldim, rdim) - 1 - i])
        for i in range(rdim - min(ldim, rdim)):
            output_shape.append(rhs.shape[rdim - min(ldim, rdim) - 1 - i])

        output_shape = list(reversed(output_shape))

        assert len(op.output) == 1
        output_name = op.output[0]
        output = Tensor(
            output_shape,
            lhs.dtype,
            layout=None,  # TODO: how to determine the layout?
            name=output_name,
            tensor_idx=output_name
        )

        matmul_op = Op.NamedOp(
            Op.OpName.MatrixOp.MatMul,
            inputs_dict,
            {
                "output": output
            },
        )

        return matmul_op.outputs


CONVERT_MAP = {
    "Add": AddConvertor,
    "AveragePool": AveragePoolConvertor,
    "Clip": ClipConvertor,
    "Concat": ConcatConvertor,
    "Constant": ConstantConvertor,
    "Conv": ConvConvertor,
    "Dropout": DropoutConvertor,
    "Flatten": FlattenConvertor,
    "Gather": GatherConvertor,
    "Gemm": GemmConvertor,
    "GlobalAveragePool": GlobalAveragePoolConvertor,
    "Identity": IdentityConvertor,
    "LRN": LRNConvertor,
    "MatMul": MatMulConvertor,
    "MaxPool": MaxPoolConvertor,
    "Mul": MulConvertor,
    "Pow": PowConvertor,
    "ReduceMean": ReduceMeanConvertor,
    "Relu": ReluConvertor,
    "Reshape": ReshapeConvertor,
    "Resize": ResizeConvertor,
    "Shape": ShapeConvertor,
    "Sigmoid": SigmoidConvertor,
    "Softmax": SoftmaxConvertor,
    "Split": SplitConvertor,
    "Transpose": TransposeConvertor,
}


class ONNXConvertor(object):
    def __init__(self, path, inference=True):
        self.path = path
        self.inference = inference

        self._tensors = {}
        self._input_names = []

    def has_tensor(self, name):
        return name in self._tensors

    def get_tensor(self, name):
        assert self.has_tensor(
            name), f"Tensor {name} is not in the convertor context."
        return self._tensors[name]

    def add_tensor(self, name, t):
        assert not self.has_tensor(name), f"Duplicated tensor {name}"
        self._tensors[name] = t

    def parse(self):
        model = onnx.load_model(self.path)
        onnx.checker.check_model(model)

        graph = model.graph

        try:
            opset_in_model = 1
            if model.opset_import:
                for opset_id in model.opset_import:
                    if str(opset_id.domain) in ["ai.onnx", ""]:
                        opset_in_model = opset_id.version
                        break
        except AttributeError:
            opset_in_model = 1

        opset = opset_in_model

        self._parse_weights(graph, opset)
        self._parse_inputs(graph, opset)

        fully_parsed = True
        for node in graph.node:
            op_name = node.op_type
            if op_name not in CONVERT_MAP:
                print(f"{op_name} is not supported yet.")
                fully_parsed = False
                break
            convertor = CONVERT_MAP[op_name](opset)
            convertor.convert(self, node)

        if not fully_parsed:
            raise RuntimeError("Parser aborts!")

        outputs = {}
        for info in graph.output:
            name, shape, dtype, shape_name = self.parse_info(info)
            t = self.get_tensor(name)
            outputs[name] = t
            assert t.shape == shape

        inputs = {}
        for name in self._input_names:
            t = self.get_tensor(name)
            inputs[name] = t

        subgraph = SubGraph(inputs, outputs)
        return Graph({"subgraph0": subgraph}, inputs, outputs)

    def parse_data(self, tensor_proto):
        np_array = to_array(tensor_proto)
        np_array = np_array.reshape(tuple(tensor_proto.dims))
        return np_array

    def parse_info(self, info_proto):
        shape = []
        shape_name = []
        for dim in info_proto.type.tensor_type.shape.dim:
            name = dim.dim_param
            value = dim.dim_value
            if not (value is not None and value > 0):
                if name == "batch_size":
                    print(
                        "Warning: batch size is not set in model file, we set it to 1.")
                    value = 1
                else:
                    raise ValueError(
                        f"tensor {info_proto.name} dim {name} value is {value}")
            shape_name.append(value)
            shape.append(value)

        name = info_proto.name
        if info_proto.type.tensor_type.elem_type:
            dtype = get_type(info_proto.type.tensor_type.elem_type)
        else:
            raise RuntimeError(f"Dtype of tensor {name} is unavailable.")

        return name, shape, dtype, shape_name

    def _parse_weights(self, graph, opset):
        for weight in graph.initializer:
            if not weight.name.strip():
                raise ValueError("Tensor's name is required in ONNX model.")
            weight_name = weight.name.strip()

            data = self.parse_data(weight)
            if self.inference:
                weight_ir = ConstTensor(
                    data.shape,
                    str(data.dtype),
                    data,
                    name=weight_name,
                    tensor_idx=weight_name
                )
            else:
                weight_ir = Tensor(
                    data.shape,
                    str(data.dtype),
                    name=weight_name,
                    tensor_idx=weight_name
                )
            self._tensors[weight_name] = weight_ir

    def _parse_inputs(self, graph, opset):
        for info in graph.input:
            name, shape, dtype, shape_name = self.parse_info(info)
            if name in self._tensors:
                raise NotImplementedError("Graph inputs duplicate.")
            self._input_names.append(name)
            self._tensors[name] = Tensor(
                shape,
                dtype,
                name=name,
                tensor_idx=name
            )

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for a in attr_proto:
            for f in ["f", "i", "s", "g"]:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ["floats", "ints", "strings"]:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ["t"]:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ["tensors"]:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ["graphs"]:
                if list(getattr(a, f)):
                    raise NotImplementedError(
                        "Field {} is not supported in relay.".format(f))
            if a.name not in attrs:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return attrs


if __name__ == "__main__":
    path = "raw_mobilenetv2.onnx"
    convertor = ONNXConvertor(path, inference=True)
    graph = convertor.parse()
    print(graph)
