import domino
import pytest
from domino.type_system import DType
from domino.program_ir import *


def test_build_expr():
    # t = DType(DTypeKind.Int, 32, 1)
    t = DType.make("int8")
    expr = Expr(t)
    assert str(expr) == "Expr(int8)"
    assert hasattr(expr, "is_const")
    assert expr.is_const() == False

def test_build_binexpr():
    t = DType.make("int8")
    a = Expr(t)
    b = Expr(t)
    c = BinExpr(t, a, b)
    assert str(c) == "BinExpr(int8, Expr(int8), Expr(int8))"

def test_build_add():
    a = ConstFloat(1.0)
    b = ConstFloat(2)
    assert a.is_const() == True
    c = Add(a, b)
    assert str(c) == "Add(float32, Const(1, float32), Const(2, float32))"

def test_build_sub():
    a = ConstFloat(1.0)
    b = ConstFloat(2)
    assert a.is_const() == True
    c = Sub(a, b)
    assert str(c) == "Sub(float32, Const(1, float32), Const(2, float32))"


if __name__ == "__main__":
    test_build_expr()
    test_build_binexpr()
    test_build_add()
    test_build_sub()