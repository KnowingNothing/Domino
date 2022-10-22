from ..base import IRBase
from ..program_ir import ConstFloat, ConstInt


class QuantParam(IRBase):
    def __init__(self) -> None:
        super(QuantParam, self).__init__()


class TensorQuantParam(QuantParam):
    pass


class OpQuantParam(QuantParam):
    pass


class ScaleQuantParam(TensorQuantParam):
    def __init__(self, scale: ConstFloat, zero_point: ConstInt) -> None:
        super(ScaleQuantParam, self).__init__()
        self.scale = scale
        self.zero_point = zero_point


class ClipQuantParam(OpQuantParam):
    def __init__(self, clip_min: ConstInt, clip_max: ConstInt) -> None:
        super(ClipQuantParam, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
