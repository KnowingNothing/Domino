import enum
from typing import Union
from xml.dom.pulldom import default_bufsize
from ..base import TypeBase


class DTypeKind(enum.Enum):
    Int = 0
    UInt = 1
    Float = 2
    BFloat = 3
    TFloat = 4
    MemRef = 5
    String = 6
    Complex = 7
    IGNORE = 254
    UNKNOWN = 255


DTYPE_KIND_TO_NAME = {
    DTypeKind.Int: "int",
    DTypeKind.UInt: "uint",
    DTypeKind.Float: "float",
    DTypeKind.BFloat: "bfloat",
    DTypeKind.TFloat: "tfloat",
    DTypeKind.MemRef: "mem_ref",
    DTypeKind.String: "string",
    DTypeKind.Complex: "complex",
    DTypeKind.IGNORE: "ignore",
    DTypeKind.UNKNOWN: "unknown"
}


class DType(TypeBase):
    """DType
    Basic scalar data type
    """

    def __init__(self, type_kind: DTypeKind, bits: int, lane: int) -> None:
        super(DType, self).__init__()
        self.type_kind = type_kind
        self.bits = bits
        self.lane = lane

    def is_int(self) -> bool:
        return self.type_kind == DTypeKind.Int

    def is_uint(self) -> bool:
        return self.type_kind == DTypeKind.UInt

    def is_float(self) -> bool:
        return self.type_kind == DTypeKind.Float

    def is_bfloat(self) -> bool:
        return self.type_kind == DTypeKind.BFloat

    def is_tfloat(self) -> bool:
        return self.type_kind == DTypeKind.TFloat

    def is_memref(self) -> bool:
        return self.type_kind == DTypeKind.MemRef

    def is_string(self) -> bool:
        return self.type_kind == DTypeKind.String

    def is_complex(self) -> bool:
        return self.type_kind == DTypeKind.Complex

    def is_ignore(self) -> bool:
        return self.type_kind == DTypeKind.IGNORE

    def is_unknown(self) -> bool:
        return self.type_kind == DTypeKind.UNKNOWN

    def copy(self) -> "DType":
        return self.__class__(self.type_kind, self.bits, self.lane)

    def with_lanes(self, lane: int) -> "DType":
        return self.__class__(self.type_kind, self.bits, lane)

    def __eq__(self, others: "DType") -> bool:
        return (self.type_kind == others.type_kind) and (self.bits == others.bits) and (self.lane == others.lane)

    def __str__(self) -> str:
        if self.lane == 1:
            return f"{DTYPE_KIND_TO_NAME[self.type_kind]}{self.bits}"
        else:
            return f"{DTYPE_KIND_TO_NAME[self.type_kind]}{self.bits}x{self.lane}"

    def __repr__(self) -> str:
        return str(self)

    def max_limit(self) -> int:
        assert self.lane == 1
        if self.is_int():
            return (1 << (self.bits - 1)) - 1
        elif self.is_uint():
            return (1 << self.bits) - 1
        else:
            raise RuntimeError("Only provide limit value for Int and UInt")

    def min_limit(self) -> int:
        assert self.lane == 1
        if self.is_int():
            return -(1 << (self.bits - 1))
        elif self.is_uint():
            return 0
        else:
            raise RuntimeError("Only provide limit value for Int and UInt")

    @classmethod
    def from_string(cls, s: str) -> "DType":
        split_pos = 0
        type_kind = DTypeKind.UNKNOWN
        default_bits = -1
        if s[:3] == "int":
            split_pos = 3
            type_kind = DTypeKind.Int
            default_bits = 32
        elif s[:4] == "uint":
            split_pos = 4
            type_kind = DTypeKind.UInt
            default_bits = 32
        elif s[:5] == "float":
            split_pos = 5
            type_kind = DTypeKind.Float
            default_bits = 32
        elif s[:6] == "bfloat":
            split_pos = 6
            type_kind = DTypeKind.BFloat
            default_bits = 16
        elif s[:6] == "tfloat":
            split_pos = 6
            type_kind = DTypeKind.TFloat
            default_bits = 32
        elif s[:7] == "mem_ref":
            split_pos = 7
            type_kind = DTypeKind.MemRef
            default_bits = 0
        elif s[:6] == "string":
            split_pos = 6
            type_kind = DTypeKind.String
            default_bits = 0
        elif s[:4] == "bool":
            split_pos = 4
            type_kind = DTypeKind.UInt
            default_bits = 1
        elif s[:7] == "complex":
            split_pos = 7
            type_kind = DTypeKind.Complex
            default_bits = 32
        elif s[:6] == "ignore":
            split_pos = 6
            type_kind = DTypeKind.IGNORE
            default_bits = 0
        elif s[:7] == "unknown":
            split_pos = 7
            type_kind = DTypeKind.UNKNOWN
            default_bits = 0
        else:
            raise RuntimeError(f"Can't parse type string {s}")

        if "x" in s[split_pos:]:
            bits, lanes = s[split_pos:].split("x")
            assert len(bits)
            assert len(lanes)
        else:
            bits = s[split_pos:]
            lanes = ""
        if len(bits):
            bits = int(bits)
        else:
            bits = default_bits
        if len(lanes):
            lanes = int(lanes)
        else:
            lanes = 1
        return cls(type_kind, bits, lanes)

    @classmethod
    def make(cls, obj: "GeneralDType") -> "DType":
        if isinstance(obj, DType):
            return obj
        return cls.from_string(obj)


GeneralDType = Union[DType, str]
