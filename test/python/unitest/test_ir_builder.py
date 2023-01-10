# from domino.program_ir import lower
from domino.program_ir import *


def test_vector_add():

    length = 20

    def vector_add(ctx, A, B, C):
        D = ctx.alloc([4], scope="local", dtype="int32", name="D")
        with ctx.spatial_for("i", range(0, length)) as i:
            with ctx.spatial_for("j", range(0, 4)) as j:
                D[j] = A[i * 4 + j] + B[i * 4 + j]
            with ctx.spatial_for("j", range(0, 4)) as j:
                C[i * 4 + j] = cast("int8", D[j] + make_const(1, "int32"))

    A = Tensor([length*4], name="A", dtype="int8")
    B = Tensor([length*4], name="B", dtype="int8")
    C = Tensor([length*4], name="C", dtype="int8")
    kernel = program_lower(vector_add, [A, B, C])
    print(kernel.signature.kernel_name)
    print(kernel.signature.kernel_args)
    print_ir(kernel.body)
    print(kernel.compiled())
    
    kernel = program_build(vector_add, [A, B, C], target="c")
    print(kernel.compiled())
    print(kernel.source)
    print(kernel.gen_signature())
    print(kernel.gen_function())
    


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
        return ret

    acc_dtype = "float32"

    def matmul(ctx, A, B, C):

        M1, N1, K1 = 4, 4, 4
        M2, N2, K2 = 2, 2, 2
        M3, N3, K3 = 2, 1, 1
        M4, N4, K4 = 16, 16, 16

        AS = ctx.alloc([M2, K2, M3, K3, M4, K4],
                       scope="shared", dtype=A.dtype, name="AS")
        BS = ctx.alloc([K2, N2, K3, N3, K4, N4],
                       scope="shared", dtype=B.dtype, name="BS")

        AR = ctx.alloc([M3, K3, M4, K4], scope="frag.A",
                       dtype=A.dtype, name="AR")
        BR = ctx.alloc([K3, N3, K4, N4], scope="frag.B",
                       dtype=B.dtype, name="BR")
        CR = ctx.alloc([M3, N3, M4, N4], scope="accumulator",
                       dtype=acc_dtype, name="CR")

        with ctx.spatial_for(names=["m1", "n1"], ranges=[range(0, M1), range(0, N1)], bindings=["blockIdx.y", "blockIdx.x"]) as (m1, n1):
            with ctx.spatial_for(names=["m2", "n2"], ranges=[range(0, M2), range(0, N2)], bindings=["threadIdx.z", "threadIdx.y"]) as (m2, n2):
                ctx.fill(CR, make_const(0, acc_dtype))
                with ctx.reduce_for(names="k1", ranges=range(0, K1)) as k1:
                    ctx.load(AS, lambda m2, k2, m3, k3, m4, k4: A[index_helper([m1, m2, m3, m4], [
                             M1, M2, M3, M4]), index_helper([k1, k2, k3, k4], [K1, K2, K3, K4])])
                    ctx.load(BS, lambda k2, n2, k3, n3, k4, n4: B[index_helper([k1, k2, k3, k4], [
                             K1, K2, K3, K4]), index_helper([n1, n2, n3, n4], [N1, N2, N3, N4])])
                    with ctx.reduce_for(names="k2", ranges=range(0, K2)) as k2:
                        ctx.load(AR, lambda m3, k3, m4,
                                 k4: AS[m2, k2, m3, k3, m4, k4])
                        ctx.load(BR, lambda k3, n3, k4,
                                 n4: BS[k2, n2, k3, n3, k4, n4])
                        with ctx.reduce_for(names="k3", ranges=range(0, K3)) as k3:
                            # No binding specified, bind to nothing, just serial loop, but not reduce loop
                            # zigzag_for is used to declare special for loop nest with at least two loop variables
                            # the iteration space is iterated in a zigzag manner.
                            # For example, zigzag([m, n], [(0, 4), (0, 2)]) is
                            # (0, 0) -> (0, 1)
                            #              |
                            #              v
                            # (1, 0) <- (1, 1)
                            #   |
                            #   v
                            # (2, 0) -> (2, 1)
                            #              |
                            #              v
                            # (3, 0) <- (3, 1)
                            with ctx.zigzag_for(names=["m3", "n3"], ranges=[range(0, M3), range(0, N3)]) as (m3, n3):
                                ctx.mma(output=CR[m3, n3, :, :], input_a=AR[m3, k3, :, :], input_b=BR[k3,
                                        n3, :, :], input_c=CR[m3, n3, :, :], layout_a="row", layout_b="row")
                ctx.store(CR, lambda m3, n3, m4, n4: C[index_helper([m1, m2, m3, m4], [
                          M1, M2, M3, M4]), index_helper([n1, n2, n3, n4], [N1, N2, N3, N4])])

        A = Tensor([M1*M2*M3*M4, K1*K2*K3*K4], "A", dtype="float16")
        B = Tensor([K1*K2*K3*K4, N1*N2*N3*N4], "B", dtype="float16")
        C = Tensor([M1*M2*M3*M4, N1*N2*N3*N4], "C", dtype="int8")
        irm = lower(matmul, [A, B, C])
        print(type(irm))
        print_ir(irm)


if __name__ == "__main__":
    test_vector_add()
