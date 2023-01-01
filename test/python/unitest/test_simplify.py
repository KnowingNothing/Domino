from domino.program_ir import *


def test_expr_simplify_match_pattern_1():
    a = Var("int32", "a")
    pattern = a * 1
    target = (Var("int32", "b") + Var("int32", "b")) * 1
    res = expr_simplify_match_pattern(target, pattern)
    assert res


def test_expr_simplify_match_pattern_2():
    pattern = Var("int32", "a") * 1
    target = Var("int32", "b") + Var("int32", "b") * 1
    res = expr_simplify_match_pattern(target, pattern)
    assert not res


def test_expr_simplify_match_pattern_3():
    a = Var("int32", "a")
    b = Var("int32", "b")
    pattern = a + b
    target = ((Var("int32", "b") + Var("int32", "b")) * 1) + \
        (Var("int32", "c") * 2 + 1)
    res = expr_simplify_match_pattern(target, pattern)
    assert res


def test_expr_simplify_match_pattern_4():
    a = Var("int32", "a")
    b = Var("int32", "b")
    pattern = a - a
    target = b - b
    res = expr_simplify_match_pattern(target, pattern)
    assert res


def test_get_expr_simplify_match_patterns_1():
    a = Var("int32", "a")
    pattern = a * 1
    target = (Var("int32", "b") + Var("int32", "b")) * 1
    res = get_expr_simplify_match_patterns(target, pattern)
    assert str(
        res) == "{Var(Const(a, string), int32): Add(int32, Var(Const(b, string), int32), Var(Const(b, string), int32))}"


def test_get_expr_simplify_match_patterns_2():
    a = Var("int32", "a")
    b = Var("int32", "b")
    c = Var("int32", "c")
    d = Var("int32", "d")
    pattern = a + b
    left = ((d + d) * 1)
    right = (c * 2 + 1)
    target = left + right

    res = get_expr_simplify_match_patterns(target, pattern)
    assert res[a] == left
    assert res[b] == right


def test_get_expr_simplify_match_patterns_3():
    a = Var("int32", "a")
    b = Var("int32", "b")
    c = Var("int32", "c")
    d = Var("int32", "d")
    pattern = a - a
    target = b - b
    res = get_expr_simplify_match_patterns(target, pattern)
    assert res[a] == b


def test_substitute_expr_1():
    a = Var("int32", "a")
    b = Var("int32", "b")
    c = Var("int32", "c")
    d = Var("int32", "d")
    expr = (((a + b) * 32 + c) * 4 + d) + 5
    mapping = {
        a: a + 1,
        b: 37,
        c: (c * c),
        d: 0
    }
    new_expr = substitute_expr(expr, mapping)
    res = print_ir(new_expr, print_out=False)
    assert res == "(((((((a + 1) + 37) * 32) + (c * c)) * 4) + 0) + 5)"


def test_simplify_expr_1():
    a = Var("int32", "a")
    b = Var("int32", "b")
    c = Var("int32", "c")
    expr = a + 0
    res = print_ir(simplify_expr(expr), print_out=False)
    assert res == "a"
    expr = 0 + b
    res = print_ir(simplify_expr(expr), print_out=False)
    assert res == "b"
    expr = ((((a + 0) * 32 + b) + 0) * 4 + c) * 2 + 0
    res = print_ir(simplify_expr(expr), print_out=False)
    assert res == "(((((a * 32) + b) * 4) + c) * 2)"


def test_simplify_expr_2():
    a = Var("int32", "a")
    b = Var("int32", "b")
    c = Var("int32", "c")
    expr = a + b + 0 - c + a * 320 * 0 + a
    res = print_ir(simplify_expr(expr), print_out=False)
    assert res == "(((a + b) - c) + a)"
    expr = (a + b + c) * 0 + c
    res = print_ir(simplify_expr(expr), print_out=False)
    assert res == "c"
    expr = a - a
    res = print_ir(simplify_expr(expr), print_out=False)
    assert res == "0"
    expr = a + b - a
    res = print_ir(simplify_expr(expr), print_out=False)
    assert res == "b"
    expr = -(-(-(-(a))))
    res = print_ir(simplify_expr(expr), print_out=False)
    assert res == "a"


if __name__ == "__main__":
    test_expr_simplify_match_pattern_1()
    test_expr_simplify_match_pattern_2()
    test_expr_simplify_match_pattern_3()
    test_expr_simplify_match_pattern_4()

    test_get_expr_simplify_match_patterns_1()
    test_get_expr_simplify_match_patterns_2()
    test_get_expr_simplify_match_patterns_3()

    test_substitute_expr_1()

    test_simplify_expr_1()
    test_simplify_expr_2()
