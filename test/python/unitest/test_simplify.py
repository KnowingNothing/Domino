from domino.program_ir import *


def test_expr_simplify_match_pattern_1():
    pattern = Var("int32", "a") * 1
    target = (Var("int32", "b") + Var("int32", "b")) * 1
    res = expr_simplify_match_pattern(target, pattern)
    assert res

def test_expr_simplify_match_pattern_2():
    pattern = Var("int32", "a") * 1
    target = Var("int32", "b") + Var("int32", "b") * 1
    res = expr_simplify_match_pattern(target, pattern)
    assert not res

def test_expr_simplify_match_pattern_3():
    pattern = Var("int32", "a") + Var("int32", "b")
    target = ((Var("int32", "b") + Var("int32", "b")) * 1) + (Var("int32", "c") * 2 + 1)
    res = expr_simplify_match_pattern(target, pattern)
    assert res


if __name__ == "__main__":
    test_expr_simplify_match_pattern_1()
    test_expr_simplify_match_pattern_2()
    test_expr_simplify_match_pattern_3()