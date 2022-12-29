# from domino.program_ir import lower
from domino.program_ir import *

def test_vector_add():

    length = 20

    def vector_add(ctx, A, B, C):
        with ctx.spatial_for("i", range(0, length)) as i:
            C[i] = A[i] + B[i]

    A = Tensor([length], "A", dtype="int8")
    B = Tensor([length], "B", dtype="int8")
    C = Tensor([length], "C", dtype="int8")
    irm = lower(vector_add, [A, B, C])
    print(irm)


def test_matmul():

    def index_helper(indices, strides):
        length = len(strides)
        assert len(indices) == length
        assert length > 0
        ret = 0
        strides = [x for x in strides]
        strides.append(1)
        for i in range(length):
            ret = (ret + indices[i]) * strides[i + 1]
        

    def matmul(ctx, A, B, C):

        M1, N1, K1 = 4, 4, 4
        M2, N2, K2 = 2, 2, 2
        M3, N3, K3 = 2, 1, 1
        M4, N4, K4 = 16, 16, 16

        AS = ctx.alloc([M2, K2, M3, K3, M4, K4], scope="shared", dtype=A.dtype, name="AS")
        BS = ctx.alloc([K2, N2, K3, N3, K4, N4], scope="shared", dtype=B.dtype, name="BS")

        AR = ctx.alloc([M3, K3, M4, K4], scope="frag.A", dtype=A.dtype, name="AR")
        BR = ctx.alloc([K3, N3, K4, N4], scope="frag.B", dtype=B.dtype, name="BR")
        CR = ctx.alloc([M3, N3, M4, N4], scope="accumulator", dtype=C.dtype, name="CR")

        with ctx.spatial_for(["m1", "n1"], [range(0, M1), range(0, N1)]) as (m1, n1):
            with ctx.reduce_for("k1", range(0, K1)) as k1:
                ctx.load(AS, A, lambda m2, k2, m3, k3, m4, k4: A[m2])


if __name__ == "__main__":
    def index_helper(indices, strides):
        length = len(strides)
        assert len(indices) == length
        assert length > 0
        ret = 0
        strides = [x for x in strides]
        strides.append(1)
        for i in range(length):
            ret = (ret + indices[i]) * strides[i + 1]
        return ret
    
    a = [Var("int32", "a"), Var("int32", "b"), Var("int32", "c")]
    print(a)
    b = [32, 8, 4]
    ret = index_helper(a, b)
    print_ir(ret)
    
