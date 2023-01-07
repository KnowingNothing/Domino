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
    res = print_ir(ndstore, print_out=False)
    assert res == "store_n((a+0), v0, v1, v2, ((v0 + 0) + 3), ((v1 + 1) + 3), ((v2 + 2) + 3));"


def test_build_store():
    a = Var("int32", "a")
    mem_ref = MemRef(a, 0)
    addr = Var("int32", "v")
    value = addr
    store = Store(mem_ref, addr, value)
    res = print_ir(store, print_out=False)
    assert res == "(a+0)[v] = v;"
    


def test_build_evaluate():
    a = Evaluate(0)
    b = Evaluate(1)
    c = Var("int32", "v")
    c = Evaluate(c // 4)
    assert str(a) == "Evaluate(Const(0, int32))"
    assert str(b) == "Evaluate(Const(1, int32))"
    assert str(c) == "Evaluate(FloorDiv(int32, Var(Const(v, string), int32), Const(4, int32)))"


if __name__ == "__main__":
    test_build_stmt()
    test_build_ndstore()
    test_build_store()
    test_build_evaluate()
