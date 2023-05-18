from domino.program_ir import *
from dominoc.ir import *
from domino.type_system import DType

t = DType.make("int32")

con_1_ = SetConst(1)
con_2_ = SetConst(2)
con_3_ = SetConst(3)

range_1 = Range(ConstInt(0, 32, 1), ConstInt(32, 32, 1), ConstInt(1, 32, 1))
range_2 = Range(ConstInt(2, 32, 1), ConstInt(3, 32, 1), ConstInt(1, 32, 1))
range_3 = Range(ConstInt(-2, 32, 1), ConstInt(5, 32, 1), ConstInt(1, 32, 1))
it_var_1 = Var(t, "x")
it_var_2 = Var(t, "y")
it_var_3 = Var(t, "z")
it_1 = Iterator(it_var_1, range_1, IterTypeKind(0))
it_2 = Iterator(it_var_2, range_2, IterTypeKind(0))
it_3 = Iterator(it_var_3, range_3, IterTypeKind(0))
con_0_x = SetConst(it_1)
con_0_y = SetConst(it_2)
assert str(con_0_x) == "SetConst(CoefNum(0), TermSet(Term(CoefNum(1), x, 1, x, x)))"
assert str(con_0_y) == "SetConst(CoefNum(0), TermSet(Term(CoefNum(1), y, 1, y, y)))"

var_a = Var(t, "a")
var_b = Var(t, "b")
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
    # 单项加减
    con_3_func = SetConstAdd(con_1_, con_2_)  # con_3_ con_2_ con_0_x con_0_y
    assert str(con_3_func) == "SetConst(CoefNum(3), TermSet())"
    con_3_x = SetConstAdd(con_3_func, con_0_x)  # con_3_x con_2_ con_0_x con_0_y
    assert str(con_3_x) == "SetConst(CoefNum(3), TermSet(Term(CoefNum(1), x, 1, x, x)))"
    con_3_x_y = SetConstAdd(con_3_x, con_0_y)  # con_3_x_y con_2_ con_0_x con_0_y
    assert (
        str(con_3_x_y)
        == "SetConst(CoefNum(3), TermSet(Term(CoefNum(1), x, 1, x, x), Term(CoefNum(1), y, 1, y, y)))"
    )
    con_3_2x_y = SetConstAdd(con_3_x_y, con_0_x)  # con_3_2x_y con_2_ con_0_x con_0_y
    assert (
        str(con_3_2x_y)
        == "SetConst(CoefNum(3), TermSet(Term(CoefNum(2), x, 1, x, x), Term(CoefNum(1), y, 1, y, y)))"
    )
    con_3_2x_y.negate()
    con_n3_n2x_ny = con_3_2x_y  # con_n3_n2x_ny con_2_ con_0_x con_0_y
    assert (
        str(con_n3_n2x_ny)
        == "SetConst(CoefNum(-3), TermSet(Term(CoefNum(-2), x, 1, x, x), Term(CoefNum(-1), y, 1, y, y)))"
    )
    con_n3_n2x = SetConstAdd(
        con_n3_n2x_ny, con_0_y
    )  # con_n3_n2x con_2_ con_0_x con_0_y
    assert (
        str(con_n3_n2x)
        == "SetConst(CoefNum(-3), TermSet(Term(CoefNum(-2), x, 1, x, x)))"
    )

    # 多项加减
    con_2_x = SetConstAdd(con_2_, con_0_x)  # con_2_x con_n3_n2x con_0_x con_0_y
    con_5_func = SetConst(5)  # con_5_ con_2_x con_n3_n2x con_0_x con_0_y
    con_5_y = SetConstAdd(
        con_5_func, con_0_y
    )  # con_5_y con_2_x con_n3_n2x con_0_x con_0_y
    con_10_5x_2y_xy = SetConstMul(
        con_2_x, con_5_y
    )  # con_10_2y_5x_xy con_5_y con_n3_n2x con_0_x con_0_y
    assert (
        str(con_10_5x_2y_xy)
        == "SetConst(CoefNum(10), TermSet(Term(CoefNum(5), x, 1, x, x), Term(CoefNum(2), y, 1, y, y), Term(CoefNum(1), x, y, 2, y, x)))"
    )
    con_0_5x_xy = SetConstMul(
        con_5_y, con_0_x
    )  # con_10_2y_5x_xy con_0_5x_xy con_n3_n2x con_0_x con_0_y
    con_n3_3x_xy = SetConstAdd(
        con_0_5x_xy, con_n3_n2x
    )  # con_n3_3x_xy con_10_2y_5x_xy con_n3_n2x con_0_x con_0_y
    assert (
        str(con_n3_3x_xy)
        == "SetConst(CoefNum(-3), TermSet(Term(CoefNum(3), x, 1, x, x), Term(CoefNum(1), y, x, 2, y, x)))"
    )
    con_n3_3x_xy.negate()
    con_3_n3x_nxy = (
        con_n3_3x_xy  # con_3_n3x_nxy con_10_2y_5x_xy con_n3_n2x con_0_x con_0_y
    )
    con_13_2x_2y = SetConstAdd(
        con_10_5x_2y_xy, con_3_n3x_nxy
    )  # con_13_2x_2y con_3_n3x_nxy con_n3_n2x con_0_x con_0_y
    assert (
        str(con_13_2x_2y)
        == "SetConst(CoefNum(13), TermSet(Term(CoefNum(2), x, 1, x, x), Term(CoefNum(2), y, 1, y, y)))"
    )

    # 乘法
    con_39_n33x_6y_n6xx_n19xy_n2xxy_n2xyy = SetConstMul(
        con_13_2x_2y, con_3_n3x_nxy
    )  # con_39_n33x_6y_n6xx_n19xy_n2xxy_n2xyy con_3_n3x_nxy con_n3_n2x con_0_x con_0_y
    assert (
        str(con_39_n33x_6y_n6xx_n19xy_n2xxy_n2xyy)
        == "SetConst(CoefNum(39), TermSet(Term(CoefNum(-33), x, 1, x, x), Term(CoefNum(6), y, 1, y, y), Term(CoefNum(-6), x, x, 2, x, x), Term(CoefNum(-19), y, x, 2, y, x), Term(CoefNum(-2), x, y, x, 3, y, x), Term(CoefNum(-2), y, y, x, 3, y, x)))"
    )
    # f = SetConstMul(con_39_n33x_6y_n6xx_n19xy_n2xxy_n2xyy, con_3_n3x_nxy) # f con_3_n3x_nxy con_n3_n2x con_0_x con_0_y
    # assert str(f) == 'SetConst(CoefNum(117), TermSet(Term(CoefNum(-216), x, 1, x, x), Term(CoefNum(18), y, 1, y, y), Term(CoefNum(81), x, x, 2, x, x), Term(CoefNum(-114), y, x, 2, y, x), Term(CoefNum(84), x, y, x, 3, y, x), Term(CoefNum(-12), y, y, x, 3, y, x), Term(CoefNum(18), x, x, x, 3, x, x), Term(CoefNum(12), x, y, x, x, 4, y, x), Term(CoefNum(25), y, y, x, x, 4, y, x), Term(CoefNum(2), x, y, x, y, x, 5, y, x), Term(CoefNum(2), y, y, x, y, x, 5, y, x)))'


def test_SetVar():
    con_n3_n2x = con_1_

    # 常系数单项加减
    # con_3_ var_0_a var_0_b
    var_3_a = SetVarAdd(var_0_a, con_3_)  # var_3_a con_3_ var_0_b
    assert (
        str(var_3_a)
        == "SetVar(SetConst(CoefNum(3), TermSet()), TermSet(Term(SetConst(CoefNum(1), TermSet()), a, 1, a, a)))"
    )
    var_9_3a = SetVarMul(var_3_a, con_3_)  # var_9_3a con_3_ var_0_b
    assert (
        str(var_9_3a)
        == "SetVar(SetConst(CoefNum(9), TermSet()), TermSet(Term(SetConst(CoefNum(3), TermSet()), a, 1, a, a)))"
    )
    var_9_3a_b = SetVarAdd(var_9_3a, var_0_b)  # var_9_3a_b con_3_ var_0_b
    assert (
        str(var_9_3a_b)
        == "SetVar(SetConst(CoefNum(9), TermSet()), TermSet(Term(SetConst(CoefNum(3), TermSet()), a, 1, a, a), Term(SetConst(CoefNum(1), TermSet()), b, 1, b, b)))"
    )
    var_9_3a_b.negate()
    var_n9_n3a_nb = var_9_3a_b  # var_n9_n3a_nb con_3_ var_0_b
    var_n9_n3a = SetVarAdd(var_n9_n3a_nb, var_0_b)  # var_n9_n3a con_3_ var_0_b
    assert (
        str(var_n9_n3a)
        == "SetVar(SetConst(CoefNum(-9), TermSet()), TermSet(Term(SetConst(CoefNum(-3), TermSet()), a, 1, a, a)))"
    )
    var_0_a_func = SetVar(var_a)
    var_0_3a = SetVarMul(var_0_a_func, con_3_)  # var_0_3a var_n9_n3a con_3_ var_0_b
    assert (
        str(var_0_3a)
        == "SetVar(SetConst(CoefNum(0), TermSet()), TermSet(Term(SetConst(CoefNum(3), TermSet()), a, 1, a, a)))"
    )
    con_n9_ = SetVarAdd(var_n9_n3a, var_0_3a)  # con_n9_ var_0_3a con_3_ var_0_b
    assert str(con_n9_) == "SetConst(CoefNum(-9), TermSet())"

    # 常系数多项加减
    # con_3_ con_n9_ var_0_3a var_0_b
    var_3_3a = SetVarAdd(var_0_3a, con_3_)
    var_n9_b = SetVarAdd(var_0_b, con_n9_)
    var_n6_3a_b = SetVarAdd(var_3_3a, var_n9_b)
    var_54_n15b_n27a_3ab_bb = SetVarMul(
        var_n9_b, var_n6_3a_b
    )  # var_54_n15b_n27a_3ab_bb var_n6_3a_b con_3_ con_n9_
    assert (
        str(var_54_n15b_n27a_3ab_bb)
        == "SetVar(SetConst(CoefNum(54), TermSet()), TermSet(Term(SetConst(CoefNum(-15), TermSet()), b, 1, b, b), Term(SetConst(CoefNum(-27), TermSet()), a, 1, a, a), Term(SetConst(CoefNum(3), TermSet()), b, a, 2, b, a), Term(SetConst(CoefNum(1), TermSet()), b, b, 2, b, b)))"
    )
    var_0_b_func = SetVar(var_b)
    var_0_n6b_3ab_bb = SetVarMul(
        var_n6_3a_b, var_0_b_func
    )  # var_54_n15b_n27a_3ab_bb var_0_n6b_3ab_bb var_0_b_func con_3_ con_n9_
    assert (
        str(var_0_n6b_3ab_bb)
        == "SetVar(SetConst(CoefNum(0), TermSet()), TermSet(Term(SetConst(CoefNum(-6), TermSet()), b, 1, b, b), Term(SetConst(CoefNum(3), TermSet()), a, b, 2, b, a), Term(SetConst(CoefNum(1), TermSet()), b, b, 2, b, b)))"
    )
    var_0_n6b_3ab_bb.negate()
    var_0_6b_n3ab_nbb = var_0_n6b_3ab_bb  # var_54_n15b_n27a_3ab_bb var_0_6b_n3ab_nbb var_0_b_func con_3_ con_n9_
    var_54_n9b_n27a = SetVarAdd(
        var_54_n15b_n27a_3ab_bb, var_0_6b_n3ab_nbb
    )  # var_54_n9b_n27a var_0_6b_n3ab_nbb var_0_b_func con_3_ con_n9_
    assert (
        str(var_54_n9b_n27a)
        == "SetVar(SetConst(CoefNum(54), TermSet()), TermSet(Term(SetConst(CoefNum(-9), TermSet()), b, 1, b, b), Term(SetConst(CoefNum(-27), TermSet()), a, 1, a, a)))"
    )

    # 非常系数加减
    # con_3_ con_n9_ con_0_x con_0_y con_n3_n2x con_39_n33x_6y_n6xx_n19xy_n2xxy_n2xyy
    # var_54_n9b_n27a var_0_6b_n3ab_nbb var_0_b_func
    con_3_2x = SetConstAdd(SetConstMul(con_0_x, SetConst(2)), con_3_)
    var_32xb = SetVarMul(var_0_b_func, con_3_2x)
    assert (
        str(var_32xb)
        == "SetVar(SetConst(CoefNum(0), TermSet()), TermSet(Term(SetConst(CoefNum(3), TermSet(Term(CoefNum(2), x, 1, x, x))), b, 1, b, b)))"
    )
    var_0_b_tmp = SetVar(var_b)
    var_0_n3n2xb = SetVarMul(var_0_b_tmp, con_n3_n2x)
    assert (
        str(var_0_n3n2xb)
        == "SetVar(SetConst(CoefNum(0), TermSet()), TermSet(Term(SetConst(CoefNum(-3), TermSet(Term(CoefNum(-2), x, 1, x, x))), b, 1, b, b)))"
    )
    con_0_ = SetVarAdd(var_0_n3n2xb, var_32xb)
    assert str(con_0_) == "SetConst(CoefNum(0), TermSet())"

    # con_3_ con_n9_ con_3_2x con_0_y con_n3_n2x con_39_n33x_6y_n6xx_n19xy_n2xxy_n2xyy
    # var_32xb var_54_n9b_n27a var_0_6b_n3ab_nbb
    con_n3y_n2xy = SetConstMul(con_n3_n2x, con_0_y)
    var_n162yn108xy_27y18xyb_81y54xya = SetVarMul(
        var_54_n9b_n27a, con_n3y_n2xy
    )  # var_n162n108x_2718xb_8154xa var_0_6b_n3ab_nbb var_0_b_func
    assert (
        str(var_n162yn108xy_27y18xyb_81y54xya)
        == "SetVar(SetConst(CoefNum(0), TermSet(Term(CoefNum(-162), y, 1, y, y), Term(CoefNum(-108), x, y, 2, y, x))), TermSet(Term(SetConst(CoefNum(0), TermSet(Term(CoefNum(27), y, 1, y, y), Term(CoefNum(18), x, y, 2, y, x))), b, 1, b, b), Term(SetConst(CoefNum(0), TermSet(Term(CoefNum(81), y, 1, y, y), Term(CoefNum(54), x, y, 2, y, x))), a, 1, a, a)))"
    )
    con_3y_2xy = SetConstMul(con_3_2x, con_0_y)
    var_18y12xyb_n9yn6xyab_n3yn2xybb = SetVarMul(var_0_6b_n3ab_nbb, con_3y_2xy)
    assert (
        str(var_18y12xyb_n9yn6xyab_n3yn2xybb)
        == "SetVar(SetConst(CoefNum(0), TermSet()), TermSet(Term(SetConst(CoefNum(0), TermSet(Term(CoefNum(18), y, 1, y, y), Term(CoefNum(12), x, y, 2, y, x))), b, 1, b, b), Term(SetConst(CoefNum(0), TermSet(Term(CoefNum(-9), y, 1, y, y), Term(CoefNum(-6), x, y, 2, y, x))), a, b, 2, b, a), Term(SetConst(CoefNum(0), TermSet(Term(CoefNum(-3), y, 1, y, y), Term(CoefNum(-2), x, y, 2, y, x))), b, b, 2, b, b)))"
    )
    # f = SetVarMul(var_n162yn108xy_27y18xyb_81y54xya, var_18y12xyb_n9yn6xyab_n3yn2xybb)
    # print(f)


if __name__ == "__main__":
    test_SetConst()
    test_SetVar()
