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
    target = ((d + d) * 1) + (c * 2 + 1)
    print("pattern:", print_ir(pattern, print_out=False))
    print("target:", print_ir(target, print_out=False))
    res = get_expr_simplify_match_patterns(target, pattern)
    for k, v in res.items():
        print(f"{print_ir(k, print_out=False)}: {print_ir(v, print_out=False)}")

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
    print_ir(new_expr)


if __name__ == "__main__":
    test_expr_simplify_match_pattern_1()
    test_expr_simplify_match_pattern_2()
    test_expr_simplify_match_pattern_3()

    test_get_expr_simplify_match_patterns_1()
    test_get_expr_simplify_match_patterns_2()

    test_substitute_expr_1()
