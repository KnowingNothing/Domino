from domino.program_ir import *
from .quantize import requantize
from .data_process import broadcast_to_s16x2, alloc_register_array, vload_to_register_array, declare_register
from .mma import mma_m2n2xk16_acc32_aoffset


def matmul_s8s8s8_acc32_m2x_n4x_k16x_row_col_mma_m2n2k16_aoffset(
        ctx, A, B, C, scales, M, N, K, input_offset, output_offset, clip_min, clip_max):
    MI = 2
    NI = 4
    KI = 16

    pack_input_offset = broadcast_to_s16x2(ctx, input_offset)
    MO, NO, KO = [ctx.map_var(name, value) for name, value in zip(
        ["MO", "NO", "KO"], [M//MI, N//NI, K//KI])]

    with ctx.spatial_for("no", Range(NO)) as no:
        scale_array = vload_to_register_array(
            ctx, "scale", scales[no*NI:(no+1)*NI], NI)
        with ctx.spatial_for("mo", Range(MO)) as mo:
            acc_array = alloc_register_array(ctx, [MI, NI], "int32", "acc", 0)
            with ctx.reduce_for("ko", Range(KO)) as ko:
                ptr_A = [A.ref(mo * 2 + i, ko * 16)
                         for i in range(MI)]
                ptr_B = [B.ref(no * 4 + i, ko * 16)
                         for i in range(NI)]
                mma_m2n2xk16_acc32_aoffset(
                    ctx, ptr_A, ptr_B, MI, NI, KI, acc_array, pack_input_offset)
            for mi in range(MI):
                for ni in range(NI):
                    C[(mo * 2 + mi), no * 4 + ni] = requantize(ctx, acc_array[mi]
                                                               [ni], scale_array[ni], output_offset, clip_min, clip_max)


def matmul_s8s8s8_acc32_m2x_n4x_k16x_row_col_mma_m2n2k16_aoffset_ring_buffer(
        ctx, A, B, C, scales, M, N, K, input_offset, output_offset, clip_min, clip_max):
    MI = 2
    NI = 4
    KI = 16

    pack_input_offset = broadcast_to_s16x2(ctx, input_offset)
    MO, NO, KO = [ctx.map_var(name, value) for name, value in zip(
        ["MO", "NO", "KO"], [M//MI, N//NI, K//KI])]

    with ctx.spatial_for("mo", Range(MO)) as mo:
        ctx.attr("ring_buffer_check_bound", C.var, C[(mo * 2), 0])
        with ctx.spatial_for("no", Range(NO)) as no:
            scale_array = vload_to_register_array(
                ctx, "scale", scales[no*NI:(no+1)*NI], NI)
            acc_array = alloc_register_array(ctx, [MI, NI], "int32", "acc", 0)
            with ctx.reduce_for("ko", Range(KO)) as ko:
                ptr_A = [A.ref(mo * 2 + i, ko * 16)
                         for i in range(MI)]
                ptr_B = [B.ref(no * 4 + i, ko * 16)
                         for i in range(NI)]
                mma_m2n2xk16_acc32_aoffset(
                    ctx, ptr_A, ptr_B, MI, NI, KI, acc_array, pack_input_offset)
            for mi in range(MI):
                for ni in range(NI):
                    C[(mo * 2 + mi), no * 4 + ni] = requantize(ctx, acc_array[mi]
                                                               [ni], scale_array[ni], output_offset, clip_min, clip_max)


def matmul_s8s8s8_acc32_aoffset_golden(
        ctx, A, B, C, scales, M, N, K, input_offset, output_offset, clip_min, clip_max):

    with ctx.spatial_for("m", Range(M)) as m:
        with ctx.spatial_for("n", Range(N)) as n:
            acc = ctx.alloc([1], scope="local", dtype="int32", name="acc")
            with ctx.reduce_for("k", Range(K)) as k:
                acc[0] = acc[0] + (cast("int32", A[m, k]) +
                                   cast("int32", input_offset)) * cast("int32", B[n, k])
            acc[0] = cast("int32", cast("float32", acc[0])
                          * scales[n]) + output_offset
            acc[0] = clip(acc[0], clip_min, clip_max)
            C[m, n] = cast("int8", acc[0])


def gen_params():
    input_offset = Var("int32", "input_offset")
    output_offset = Var("int32", "output_offset")
    clip_max = Var("int32", "clip_max")
    clip_min = Var("int32", "clip_min")
    M = Var("int32", "M")
    N = Var("int32", "N")
    K = Var("int32", "K")
    A = Tensor([M, K], name="A", dtype="int8")
    B = ConstTensor([N, K], name="B", dtype="int8")
    C = Tensor([M, N], name="C", dtype="int8")
    scales = ConstTensor([N], name="scales", dtype="float32")
    return [A, B, C, scales], [M, N, K, input_offset, output_offset, clip_min, clip_max]


def gen_matmul():
    tensors, scalars = gen_params()
    kernel = program_build(matmul_s8s8s8_acc32_m2x_n4x_k16x_row_col_mma_m2n2k16_aoffset,
                           tensors, scalar_inputs=scalars, target="arm_m")
    return kernel, tensors, scalars


def gen_matmul_ring_buffer():
    tensors, scalars = gen_params()
    # print_ir(program_lower(matmul_s8s8s8_acc32_m2x_n4x_k16x_row_col_mma_m2n2k16_aoffset_ring_buffer,
    #                       tensors, scalar_inputs=scalars))
    kernel = program_build(matmul_s8s8s8_acc32_m2x_n4x_k16x_row_col_mma_m2n2k16_aoffset_ring_buffer,
                           tensors, scalar_inputs=scalars, target="arm_m")
    return kernel, tensors, scalars


def gen_matmul_golden():
    tensors, scalars = gen_params()

    kernel = program_build(matmul_s8s8s8_acc32_aoffset_golden,
                           tensors, scalar_inputs=scalars, target="arm_m")

    return kernel, tensors, scalars
