from typing import Union, Any
from ..base import IRBase
from ..type_system.dtype import DType
from ..type_system.ttype import ShapeType, TType
from .op_base import OpBase


class Tensor(IRBase):
    def __init__(
            self,
            shape: ShapeType,
            dtype: Union[DType, str],
            produce_op: OpBase,
            out_idx: int,
            layout: str,
            name: str = "tensor"):
        super(Tensor, self).__init__()
        self.ttype: TType = TType(dtype, shape)
        self.produce_op: OpBase = produce_op
        self.out_idx: int = out_idx
        self.layout_str: str = layout
        self.name: str = name

    @property
    def shape(self) -> ShapeType:
        return self.ttype.shape

    @property
    def dtype(self) -> DType:
        return self.ttype.dtype

    @property
    def layout(self) -> str:
        return self.layout_str

    def __str__(self) -> str:
        return f"Tensor({self.ttype}, {self.name}, {self.layout}, {self.produce_op})"

    def __repr__(self) -> str:
        return str(self)
