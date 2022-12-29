from domino.program_ir import *


def test_build_stmt():
    a = Stmt()
    assert str(a) == "Stmt()"


def test_build_ndstore():
    a = Var("int32", "a")
    mem_ref = MemRef(a, 0)
    indices = [Var("int32", f"v{i}") for i in range(3)]
    values = [indices[i] + i + 3 for i in range(3)]
    ndstore = NdStore(mem_ref, indices, values)
    assert str(ndstore) == (
        "NdStore(MemRef(Var(Const(a, string0), int32), "
        "Const(0, int32)), ExprList(Var(Const(v0, string0), int32), "
        "Var(Const(v1, string0), int32), Var(Const(v2, string0), int32)), "
        "ExprList(Add(int32, Add(int32, Var(Const(v0, string0), int32), Const(0, int32)), "
        "Const(3, int32)), Add(int32, Add(int32, Var(Const(v1, string0), int32), Const(1, int32)), "
        "Const(3, int32)), Add(int32, Add(int32, Var(Const(v2, string0), int32), Const(2, int32)), Const(3, int32))))")


def test_build_store():
    a = Var("int32", "a")
    mem_ref = MemRef(a, 0)
    addr = Var("int32", "v")
    value = addr
    store = Store(mem_ref, addr, value)
    assert str(store) == (
        "Store(MemRef(Var(Const(a, string0), int32), Const(0, int32)), "
        "Var(Const(v, string0), int32), Var(Const(v, string0), int32))")


def test_build_evaluate():
    a = Evaluate(0)
    b = Evaluate(1)
    c = Var("int32", "v")
    c = Evaluate(c // 4)
    assert str(a) == "Evaluate(Const(0, int32))"
    assert str(b) == "Evaluate(Const(1, int32))"
    assert str(c) == "Evaluate(FloorDiv(int32, Var(Const(v, string0), int32), Const(4, int32)))"


if __name__ == "__main__":
    test_build_stmt()
    test_build_ndstore()
    test_build_store()
    test_build_evaluate()
