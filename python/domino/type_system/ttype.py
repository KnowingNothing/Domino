import enum
from typing import Union, List, Optional, Tuple
from ..base import TypeBase
from ..program_ir.scalar_expr import ExprBase
from .dtype import DType, GeneralDType


ShapeType = Union[List[Union[int, ExprBase]], Tuple[Union[int, ExprBase]]]


class TType(TypeBase):
    """TType
    Tensor Type: dtype + shape
    """

    def __init__(
            self,
            dtype: GeneralDType,
            shape: ShapeType) -> None:
        super(TType, self).__init__()
        self.dtype = DType.make(dtype)
        self.shape = shape

    def __str__(self) -> str:
        return f"[{','.join(map(str, self.shape))}]x{self.dtype}"

    def __repr__(self) -> str:
        return str(self)
