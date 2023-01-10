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
    # shape constants
    K = 16

    # Quantize constants
    CLIP_MIN = 0
    CLIP_MAX = 255

    # optimization parameters
    KI = 4
    NI = 2
    MI = 2

    # this kernel is designed for K=16, so KO is known at compile time
    KO = K // KI

    def matmul_s8_s16_k16(ctx, A, B, C, scales, MO, NO, input_offset, output_offset):
        vA = ctx.alloc([MI, KI//2], scope="local", dtype="int32", name="vA")
        vB = ctx.alloc([NI, KI//2], scope="local", dtype="int32", name="vB")
        scale = ctx.alloc([NI], scope="local", dtype="float32", name="scale")
        input_offset_s16 = ctx.map_var(
            "input_offset_s16", cast("int16", input_offset))
        pack_input_offset = ctx.map_var("pack_input_offset", pack_value(
            "int32", [input_offset_s16, input_offset_s16]))
        with ctx.spatial_for("mo", Range(MO)) as mo:
            with ctx.spatial_for("no", Range(NO)) as no:
                acc = ctx.alloc([MI, NI], scope="local",
                                dtype="int32", name="acc")
                with ctx.unroll_for("ni", Range(NI)) as ni:
                    scale[ni] = scales[no * NI + ni]
                # with ctx.pipeline_for("ko", Range(KO)) as ko:
                #     # load 4 int8 and extend to 2 int 32
                #     # high addr <-> low addr
                #     # [v1](8bit), [v2](8bit), [v3](8bit), [v4](8bit)
                #     # -> [[v2](16bit), [v4](16bit)](32bit), [[v1](16bit), [v3](16bit)](32bit)
                #     ctx.load_s8x4_s32x2_ext(vA[0, :], A[mo * MI, ko * KI:KI])
                #     ctx.s16_vadd_s32x2_s32_s32x2(vA[0, :], vA[0, :], pack_input_offset)
                #     ctx.pipeline_slot()
                #     ctx.load_s8x4_s32x2_ext(vB[0, :], B[no * NI, ko * KI:KI])
                #     ctx.s16_dot_s32x2_s32x2_s32(vA[0, :], vB[0, :], acc[0, 0])
                #     ctx.load_s8x4_s32x2_ext(vA[1, :], A[mo * MI + 1, ko * KI:KI])
                #     ctx.s16_vadd_s32x2_s32_s32x2(vA[1, :], vA[1, :], pack_input_offset)
                #     ctx.s16_dot_s32x2_s32x2_s32(vA[1, :], vB[0, :], acc[1, 0])
                #     ctx.load_s8x4_s32x2_ext(vB[1, :], B[mo * MI + 1, ko * KI:KI])
                #     ctx.s16_dot_s32x2_s32x2_s32(vA[0, :], vB[1, :], acc[0, 1])
                #     ctx.pipeline_slot()
                #     ctx.s16_dot_s32x2_s32x2_s32(vA[1, :], vB[1, :], acc[1, 1])
                # with ctx.unroll_for("mi", Range(MI)) as mi:
                #     with ctx.unroll_for("ni", Range(NI)) as ni:
                #         min_value = make_const(CLIP_MIN, acc.dtype)
                #         max_value = make_const(CLIP_MAX, acc.dtype)
                #         value = cast(acc.dtype, (cast("float32", acc[mi, ni]) * scale[ni])) + cast(acc.dtype, output_offset)
                #         C[mo * MI + mi, no * NI + ni] = ctx.clip(C.dtype, value, min=min_value, max=max_value)

    input_offset = Var("int32", "input_offset")
    output_offset = Var("int32", "output_offset")
    MO = Var("int32", "MO")
    NO = Var("int32", "NO")
    A = Tensor([MO*MI, K], name="A", dtype="int8")
    B = Tensor([NO*NI, K], name="B", dtype="int8")
    C = Tensor([MO*MI, NO*NI], name="C", dtype="int8")
    scales = Tensor([NO*NI], name="scales", dtype="float32")

    kernel = program_lower(matmul_s8_s16_k16, [A, B, C, scales], scalar_inputs=[
                           MO, NO, input_offset, output_offset])
    print_ir(kernel.body)

    kernel = program_build(matmul_s8_s16_k16, [A, B, C, scales], scalar_inputs=[
                           MO, NO, input_offset, output_offset], target="arm_m")
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
