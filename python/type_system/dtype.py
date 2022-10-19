import enum
from ..base import TypeBase


class DTypeKind(enum.Enum):
    Int = 0
    UInt = 1
    Float = 2
    BFloat = 3
    TFloat = 4
    MemRef = 5
    IGNORE = 6
    UNKNOWN = 255


DTYPE_KIND_TO_NAME = {
    DTypeKind.Int: "int",
    DTypeKind.UInt: "uint",
    DTypeKind.Float: "float",
    DTypeKind.BFloat: "bfloat",
    DTypeKind.TFloat: "tfloat",
    DTypeKind.MemRef: "mem_ref",
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

    @classmethod
    def from_string(cls, s: str) -> "DType":
        split_pos = 0
        type_kind = DTypeKind.UNKNOWN
        if s[:3] == "int":
            split_pos = 3
            type_kind = DTypeKind.Int
        elif s[:4] == "uint":
            split_pos = 4
            type_kind = DTypeKind.UInt
        elif s[:5] == "float":
            split_pos = 5
            type_kind = DTypeKind.Float
        elif s[:6] == "bfloat":
            split_pos = 6
            type_kind = DTypeKind.BFloat
        elif s[:6] == "tfloat":
            split_pos = 6
            type_kind = DTypeKind.TFloat
        elif s[:7] == "mem_ref":
            split_pos = 7
            type_kind = DTypeKind.MemRef
        elif s[:6] == "ignore":
            split_pos = 6
            type_kind = DTypeKind.IGNORE
        elif s[:7] == "unknown":
            split_pos = 7
            type_kind = DTypeKind.UNKNOWN
        else:
            raise RuntimeError(f"Can't parse type string {s}")

        bits, lanes = s[split_pos:].split("x")
        bits = int(bits)
        if len(lanes):
            lanes = int(lanes)
        else:
            lanes = 1
        return cls(type_kind, bits, lanes)
