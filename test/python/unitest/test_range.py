from domino.program_ir import *
from dominoc.ir import *
from domino.type_system import DType

t = DType.make("int32")

con_0 = ConstInt(0, 32, 1)
con_1 = ConstInt(1, 32, 1)
con_n1 = ConstInt(-1, 32, 1)
con_2 = ConstInt(2, 32, 1)
con_n2 = ConstInt(-2, 32, 1)
con_3 = ConstInt(3, 32, 1)
con_n3 = ConstInt(-3, 32, 1)

con_2_ = SetConst(2)
con_3_ = SetConst(3)

range_x = Range(ConstInt(0, 32, 1), ConstInt(32, 32, 1), ConstInt(1, 32, 1))
range_y = Range(ConstInt(2, 32, 1), ConstInt(3, 32, 1), ConstInt(1, 32, 1))
range_z0 = Range(ConstInt(0, 32, 1), ConstInt(2, 32, 1), ConstInt(1, 32, 1))
it_var_x = Var(t, "x")
it_var_y = Var(t, "y")
it_var_z0 = Var(t, "z0")
it_x = Iterator(it_var_x, range_x, IterTypeKind(0))
it_y = Iterator(it_var_y, range_y, IterTypeKind(0))
it_z0 = Iterator(it_var_z0, range_z0, IterTypeKind(0))
con_0_x = SetConst(it_x)
con_0_y = SetConst(it_y)
assert str(con_0_x) == "SetConst(CoefNum(0), TermSet(Term(CoefNum(1), x, 1, x, x)))"
assert str(con_0_y) == "SetConst(CoefNum(0), TermSet(Term(CoefNum(1), y, 1, y, y)))"

var_a = Var(t, "a")
var_b = Var(t, "b")
var_c0 = Var(t, "c0")
var_0_a = SetVar(var_a)
var_0_b = SetVar(var_b)
assert (
    str(var_0_a)
    == "SetVar(SetConst(CoefNum(0), TermSet()), TermSet(Term(SetConst(CoefNum(1), TermSet()), a, 1, a, a)))"
)
assert (
    str(var_0_b)
    == "SetVar(SetConst(CoefNum(0), TermSet()), TermSet(Term(SetConst(CoefNum(1), TermSet()), b, 1, b, b)))"
)


def test_SetConst():
    assert (
        str(HandleSetConst(con_2_))
        == "Range(Const(2, int32), Const(1, int32), Const(1, int32))"
    )
    SetConstAdd(SetConstMul(con_0_x, con_2_), SetConstMul(con_3_, con_0_y)).negate()
    con_2_n2x_n3y = SetConstAdd(con_0_x, con_2_)
    assert (
        str(HandleSetConst(con_2_n2x_n3y))
        == "Range(Const(-72, int32), Const(69, int32), Const(1, int32))"
    )
    con_2y_n2xy_n3yy = SetConstMul(con_2_n2x_n3y, con_0_y)
    assert (
        str(HandleSetConst(con_2y_n2xy_n3yy))
        == "Range(Const(-292, int32), Const(289, int32), Const(1, int32))"
    )


def test_SetVar():
    con_2_.negate()
    con_n2_ = con_2_
    var_n2_n2a = SetVarAdd(SetVarMul(var_0_a, con_n2_), con_n2_)
    assert (
        str(HandleSetVar(var_n2_n2a))
        == "Range(Add(int32, Const(-2, int32), Mul(int32, Var(a, int32), Const(-2, int32))), Const(1, int32), Const(1, int32))"
    )
    assert (
        str(HandleSetVar(var_0_b))
        == "Range(Var(b, int32), Const(1, int32), Const(1, int32))"
    )
    var_0_b.negate()
    var_0_nb = var_0_b
    assert (
        str(HandleSetVar(var_0_nb))
        == "Range(Neg(int32, Var(b, int32)), Const(1, int32), Const(1, int32))"
    )
    var_n12yb_2yab = SetVarAdd(
        SetVarMul(SetVarMul(var_n2_n2a, var_0_nb), con_0_y), var_0_nb
    )
    assert (
        str(HandleSetVar(var_n12yb_2yab))
        == "Range(Add(int32, Mul(int32, Var(b, int32), Const(3, int32)), Mul(int32, Mul(int32, Var(a, int32), Var(b, int32)), Const(4, int32))), Add(int32, Add(int32, Const(1, int32), Mul(int32, Var(b, int32), Const(4, int32))), Mul(int32, Mul(int32, Var(a, int32), Var(b, int32)), Const(4, int32))), Const(1, int32))"
    )


import random
import time
import math
from itertools import product

MaxHeight = 50
LeafList = [con_0, con_1, con_n1, it_x, it_y, var_a, var_b]
LeafDivident, OpDivident = 1 / len(LeafList), 1 / 4


def generate_expr_tree(height):
    if height > MaxHeight:
        rand_num = random.random()
        return LeafList[math.floor(rand_num / LeafDivident)]
    rand_num = random.random() * 2
    if rand_num < 1:
        return LeafList[math.floor(rand_num / LeafDivident)]
    Op = math.floor((rand_num - 1) / OpDivident)
    if Op == 0:
        return Add(generate_expr_tree(height + 1), generate_expr_tree(height + 1))
    if Op == 1:
        return Sub(generate_expr_tree(height + 1), generate_expr_tree(height + 1))
    if Op == 2:
        return Mul(generate_expr_tree(height + 1), generate_expr_tree(height + 1))
    return Neg(generate_expr_tree(height + 1))


ComputeBeg, ComputeExtent = 0, 1


def test_Total():
    root = generate_expr_tree(0)
    # print(root)
    myRange = InferBound(root)
    # print(myRange)
    # var_a var_b var_c0取不同值时：挑it算root然后看是否在myRange以内
    for a, b, c0 in product(range(5), repeat=3):
        my_dict = {"a": a, "b": b, "c0": c0}
        lower_bound = compute_range(myRange, my_dict, ComputeBeg)
        upper_bound = lower_bound + compute_range(myRange, my_dict, ComputeExtent) - 1
        # print(a, b, c0)
        # print('[', lower_bound, upper_bound, ']')
        for x, y, z0 in product([0, 1, 5, 30, 31], [2, 3, 4], [0, 1]):
            my_dict.update({"x": x, "y": y, "z0": z0})
            value = compute_expr(root, my_dict)
            # print(x, y, z0)
            # print('(', value, ')')
            assert lower_bound <= value <= upper_bound


if __name__ == "__main__":
    test_SetConst()
    test_SetVar()
    # 连带着测transform.h
    for t in range(100):
        random.seed(time.time())
        test_Total()
