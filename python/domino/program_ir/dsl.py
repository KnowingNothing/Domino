from ..type_system import DType
from .scalar_expr import Expr, Var, ConstVar
from typing import List, Union


__all__ = ["Tensor", "ConstTensor"]


class Tensor(object):
    def __init__(
            self,
            shape: List[Union[int, Expr]],
            name: str = "",
            dtype: Union[DType, str] = "float32"):
        self.shape = shape
        self.name = name
        self.dtype = dtype

    def is_const(self):
        return False

    @property
    def var(self):
        return Var(self.dtype, self.name)


class ConstTensor(Tensor):
    def __init__(
            self,
            shape: List[Union[int, Expr]],
            name: str = "",
            dtype: Union[DType, str] = "float32"):
        super(ConstTensor, self).__init__(shape, name, dtype)

    def is_const(self):
        return True

    @property
    def var(self):
        return ConstVar(self.dtype, self.name)
