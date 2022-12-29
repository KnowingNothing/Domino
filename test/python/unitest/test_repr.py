from domino.program_ir import *


def test_print_expr():
    a = Var("int32", "v")
    res = print_ir(a)
    assert res == "v"

    a = a + 2
    res = print_ir(a)
    assert res == "(v + 2)"


if __name__ == "__main__":
    test_print_expr()