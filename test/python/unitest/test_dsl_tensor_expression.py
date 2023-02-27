from domino.program_ir import *


def test_te_loop():
    m = SLoop(32, "m")
    n = SLoop(32, "n")
    k = RLoop(32, "k")
    m0 = TLoop(2, "m")
    n0 = TLoop(2, "n")
    k0 = TLoop(16, "k")
    # print(m, n, k)
    # print(m0, n0, k0)


def test_te_tensor_access():
    m = SLoop(32, "m")
    n = SLoop(32, "n")
    A = Tensor([32, 32], name="A", dtype="int8")
    xx = A[m:m+4, n:n+4][1:2, ::2][0,0]
    # print(xx)

def test_te_tensor_compute():
    m = SLoop(32, "m")
    n = SLoop(32, "n")
    k = RLoop(32, "k")
    A = Tensor([32, 32], name="A", dtype="int8")
    B = Tensor([32, 32], name="B", dtype="int8")
    C = Tensor([32, 32], name="C", dtype="int8")
    xx = A[m, k] * B[k, n]
    print_ir(xx)


if __name__ == "__main__":
    test_te_loop()
    test_te_tensor_access()
    test_te_tensor_compute()