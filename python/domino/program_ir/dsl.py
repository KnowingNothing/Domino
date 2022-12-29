from ..type_system import DType
from .scalar_expr import Expr
from typing import List, Union

class Tensor(object):
    def __init__(self, shape: List[Union[int, Expr]], name: str, dtype: Union[DType, str] = "float32") -> None:
        self.shape = shape
        self.name = name