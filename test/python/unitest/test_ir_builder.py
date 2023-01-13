# from domino.program_ir import lower
from domino.program_ir import *


def test_vector_add_static_shape_c():

    length = 20

    def vector_add_static_shape(ctx, A, B, C):
        D = ctx.alloc([4], scope="local", dtype="int32", name="D")
        with ctx.spatial_for("i", range(0, length)) as i:
            with ctx.spatial_for("j", range(0, 4)) as j:
                D[j] = A[i * 4 + j] + B[i * 4 + j]
            with ctx.spatial_for("j", range(0, 4)) as j:
                C[i * 4 + j] = cast("int8", D[j] + make_const(1, "int32"))

    A = Tensor([length*4], name="A", dtype="int8")
    B = Tensor([length*4], name="B", dtype="int8")
    C = Tensor([length*4], name="C", dtype="int8")
    kernel = program_lower(vector_add_static_shape, [A, B, C])
    print(kernel.signature.kernel_name)
    print(kernel.signature.kernel_args)
    print_ir(kernel.body)
    print(kernel.compiled())

    kernel = program_build(vector_add_static_shape, [A, B, C], target="c")
    print(kernel.compiled())
    print(kernel.source)
    print(kernel.gen_signature())
    print(kernel.gen_function())


def test_vector_add_dynamic_shape_c():

    def vector_add_dynamic_shape(ctx, A, B, C, length):
        D = ctx.alloc([4], scope="local", dtype="int32", name="D")
        with ctx.spatial_for("i", Range(0, length)) as i:
            with ctx.spatial_for("j", range(0, 4)) as j:
                D[j] = A[i * 4 + j] + B[i * 4 + j]
            with ctx.spatial_for("j", range(0, 4)) as j:
                C[i * 4 + j] = cast("int8", D[j] + make_const(1, "int32"))

    length = Var("int32", "length")
    A = Tensor([length*4], name="A", dtype="int8")
    B = Tensor([length*4], name="B", dtype="int8")
    C = Tensor([length*4], name="C", dtype="int8")
    kernel = program_lower(vector_add_dynamic_shape, [
                           A, B, C], scalar_inputs=[length])
    print(kernel.signature.kernel_name)
    print(kernel.signature.kernel_args)
    print_ir(kernel.body)
    print(kernel.compiled())

    kernel = program_build(vector_add_dynamic_shape, [
                           A, B, C], scalar_inputs=[length], target="c")
    print(kernel.compiled())
    print(kernel.source)
    print(kernel.gen_signature())
    print(kernel.gen_function())


def test_matmul_dynamic_shape_mcu():

    # Quantize constants
    CLIP_MIN = 0
    CLIP_MAX = 255

    # optimization parameters
    NI = 2
    MI = 2

    def matmul_s8s8s8_acc32_mx_n4x_k16x_row_col_mma_m2n2k16_aoffset(
            ctx, A, B, C, scales, M, N, K, input_offset, output_offset):
        # vA = ctx.alloc([MI, KI//2], scope="local", dtype="int32", name="vA")
        # vB = ctx.alloc([NI, KI//2], scope="local", dtype="int32", name="vB")
        # scale = ctx.alloc([NI], scope="local", dtype="float32", name="scale")
        input_offset_s16 = ctx.map_var(
            "input_offset_s16", cast("int16", input_offset))
        pack_input_offset = ctx.map_var("pack_input_offset", pack_value(
            "int32", [input_offset_s16, input_offset_s16]))
        MO = ctx.map_var("MO", M//2)
        NO = ctx.map_var("NO", N//4)
        KO = ctx.map_var("KO", K//16)
        with ctx.spatial_for("no", Range(NO)) as no:
            scale0 = ctx.map_var("scale0", scales[no])
            scale1 = ctx.map_var("scale1", scales[no + 1])
            scale2 = ctx.map_var("scale1", scales[no + 2])
            scale3 = ctx.map_var("scale1", scales[no + 3])
            with ctx.spatial_for("mo", Range(MO)) as mo:
                acc00 = ctx.map_var("acc00", make_const(0, "int32"))
                acc01 = ctx.map_var("acc01", make_const(0, "int32"))
                acc02 = ctx.map_var("acc02", make_const(0, "int32"))
                acc03 = ctx.map_var("acc03", make_const(0, "int32"))
                acc10 = ctx.map_var("acc10", make_const(0, "int32"))
                acc11 = ctx.map_var("acc11", make_const(0, "int32"))
                acc12 = ctx.map_var("acc12", make_const(0, "int32"))
                acc13 = ctx.map_var("acc13", make_const(0, "int32"))
                with ctx.reduce_for("ko", Range(KO)) as ko:
                    ctx.call("ignore", "mma_m2n2k16_s8s8s8_acc32_aoffset", (
                        MemRef(A.var, mo * 2 * K + ko *
                               16), MemRef(B.var, no * 4 * K + ko * 16),
                        MemRef(A.var, (mo + 1) * 2 * K + ko *
                               16), MemRef(B.var, (no + 1) * 4 * K + ko * 16),
                        pack_input_offset, acc00, acc01, acc10, acc11
                    ))
                    ctx.call("ignore", "mma_m2n2k16_s8s8s8_acc32_aoffset", (
                        MemRef(A.var, mo * 2 * K + ko *
                               16), MemRef(B.var, (no + 2) * 4 * K + ko * 16),
                        MemRef(A.var, (mo + 1) * 2 * K + ko *
                               16), MemRef(B.var, (no + 3) * 4 * K + ko * 16),
                        pack_input_offset, acc02, acc03, acc12, acc13
                    ))
                C[(mo * 2) * N + no * 4] = cast("int8", cast("int32",
                                         (cast("float", acc00) * scale0)) + output_offset)
                C[(mo * 2) * N + no * 4 + 1] = cast("int8", cast("int32",
                                             (cast("float", acc01) * scale1)) + output_offset)
                C[(mo * 2) * N + no * 4 + 2] = cast("int8", cast("int32",
                                             (cast("float", acc02) * scale2)) + output_offset)
                C[(mo * 2) * N + no * 4 + 3] = cast("int8", cast("int32",
                                             (cast("float", acc03) * scale3)) + output_offset)
                C[(mo * 2 + 1) * N + no * 4] = cast("int8", cast("int32",
                                             (cast("float", acc10) * scale0)) + output_offset)
                C[(mo * 2 + 1) * N + no * 4 + 1] = cast("int8", cast(
                    "int32", (cast("float", acc11) * scale1)) + output_offset)
                C[(mo * 2 + 1) * N + no * 4 + 2] = cast("int8", cast(
                    "int32", (cast("float", acc12) * scale2)) + output_offset)
                C[(mo * 2 + 1) * N + no * 4 + 3] = cast("int8", cast(
                    "int32", (cast("float", acc13) * scale3)) + output_offset)

    input_offset = Var("int32", "input_offset")
    output_offset = Var("int32", "output_offset")
    M = Var("int32", "M")
    N = Var("int32", "N")
    K = Var("int32", "K")
    A = Tensor([M, K], name="A", dtype="int8")
    B = Tensor([N, K], name="B", dtype="int8")
    C = Tensor([M, N], name="C", dtype="int8")
    scales = Tensor([N], name="scales", dtype="float32")

    kernel = program_lower(matmul_s8s8s8_acc32_mx_n4x_k16x_row_col_mma_m2n2k16_aoffset, [A, B, C, scales], scalar_inputs=[
                           M, N, K, input_offset, output_offset])
    print_ir(kernel.body)

    kernel = program_build(matmul_s8s8s8_acc32_mx_n4x_k16x_row_col_mma_m2n2k16_aoffset, [A, B, C, scales], scalar_inputs=[
                           M, N, K, input_offset, output_offset], target="arm_m")
    print(kernel.source)
    print(kernel.gen_function())


def test_matmul_gpu():

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
    # test_vector_add_dynamic_shape_c()
    test_matmul_dynamic_shape_mcu()
