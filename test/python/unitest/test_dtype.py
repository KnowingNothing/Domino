import dominoc
from dominoc import DType, DTypeKind


def test_build_dtype():
    t = dominoc.DType(DTypeKind.Int, 32, 1)
    print(t)


if __name__ == "__main__":
    test_build_dtype()