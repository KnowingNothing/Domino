from typing import Union, Any, Optional
import numpy as np
from ..base import IRBase
from ..type_system.dtype import DType, GeneralDType
from ..type_system.ttype import ShapeType, TType
from .op_base import OpBase
from .quantize import TensorQuantParam, ScaleQuantParam, ClipQuantParam


class Tensor(IRBase):
    def __init__(
            self,
            shape: ShapeType,
            dtype: GeneralDType,
            produce_op: Optional[OpBase] = None,
            out_idx: str = "",
            layout: Optional[str] = None,
            name: str = "tensor",
            quant_params: Optional[TensorQuantParam] = None,
            tensor_idx: Union[int, str] = 0):
        super(Tensor, self).__init__()
        self.ttype: TType = TType(dtype, shape)
        self.produce_op: Optional[OpBase] = produce_op
        self.out_idx: str = out_idx
        self.layout_str: Optional[str] = layout
        self.name: str = name
        self.quant_params: Optional[TensorQuantParam] = quant_params
        self.tensor_idx: Union[int, str] = tensor_idx

    @property
    def shape(self) -> ShapeType:
        return [x for x in self.ttype.shape]

    @property
    def dtype(self) -> DType:
        return self.ttype.dtype

    @dtype.setter
    def dtype(self, new_dtype: GeneralDType):
        self.ttype = TType(new_dtype, self.shape)

    @property
    def layout(self) -> Optional[str]:
        return self.layout_str

    @layout.setter
    def layout(self, layout_str: str):
        self.layout_str = layout_str

    @property
    def shape_dict(self):
        return {l: s for s, l in zip(self.shape, self.layout)}

    def __str__(self) -> str:
        return f"Tensor({self.ttype}, {self.name}, {self.layout}, {hex(id(self.produce_op))})"

    def __repr__(self) -> str:
        return f"Tensor({self.ttype}, {self.name})"


class ConstTensor(Tensor):
    def __init__(
            self,
            shape: ShapeType,
            dtype: GeneralDType,
            value: np.array,
            layout: Optional[str] = None,
            name: str = "tensor",
            quant_params: Optional[TensorQuantParam] = None,
            tensor_idx: Union[int, str] = 0):
        super(ConstTensor, self).__init__(
            shape, dtype, None, "", layout, name, quant_params, tensor_idx)
        self._value = value

    @property
    def value(self):
        return self._value


class Attribute(IRBase):
    def __init__(self, value: IRBase) -> None:
        super(Attribute, self).__init__()
        assert isinstance(
            value, IRBase), "Please only use IRBase objects for Attribute"
        self.value = value
