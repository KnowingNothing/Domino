from domino.program_ir import *
from .quantize import requantize
from .data_process import broadcast_to_s16x2, alloc_register_array, vload_to_register_array
from .mma import mma_m2n2xk16_acc32_aoffset


def batch_matmul_chain_s8s8s8_acc32_m2x_n4x_k16x_l16x_row_col_col_mma_m2n2k16_aoffset(
        ctx, A, B, C, D, inter_scales, scales, batch, M, N, K, L, input_offset, inter_offset1, inter_offset2, output_offset, clip_min, clip_max):
    MI = 2
    NI = 4
    KI = 16

    pack_input_offset = broadcast_to_s16x2(ctx, input_offset)
    pack_inter_offset = broadcast_to_s16x2(ctx, inter_offset2)
    MO, NO, LO, KO = [ctx.map_var(name, value) for name, value in zip(
        ["MO", "NO", "LO", "KO"], [M//MI, N//NI, L//KI, K//KI])]

    with ctx.spatial_for("batch", Range(batch)) as batch:
        with ctx.spatial_for("mo", Range(MO)) as mo:
            inter = ctx.alloc([MI, KI], scope="local",
                              dtype="int8", name="inter")
            acc2_array = alloc_register_array(
                ctx, [MI, NI], "int32", "acc2", 0)
            with ctx.reduce_for("lo", Range(LO)) as lo:
                with ctx.spatial_for("li", Range(KI//NI)) as li:
                    inter_scale_array = vload_to_register_array(
                        ctx, "inter_scale", inter_scales[batch*L:(batch+1)*L][lo*KI:(lo+1)*KI][li*NI:(li+1)*NI], NI)
                    acc1_array = alloc_register_array(
                        ctx, [MI, NI], "int32", "acc1", 0)
                    with ctx.reduce_for("ko", Range(KO)) as ko:
                        ptr_A = [MemRef(A.var, batch * M * K + (mo * MI + mi) * K + ko * KI)
                                 for mi in range(MI)]
                        ptr_B = [MemRef(B.var, batch * N * K + (lo * KI + li * NI + ni) * K + ko * KI)
                                 for ni in range(NI)]
                        mma_m2n2xk16_acc32_aoffset(
                            ctx, ptr_A, ptr_B, MI, NI, KI, acc1_array, pack_input_offset)
                    for mi in range(MI):
                        for ni in range(NI):
                            inter[mi, li * NI + ni] = requantize(
                                ctx, acc1_array[mi][ni], inter_scale_array[ni], inter_offset1, clip_min, clip_max)

                with ctx.spatial_for("no", Range(NO)) as no:
                    scale_array = vload_to_register_array(
                        ctx, "scale", scales[batch*N:(batch+1)*N][no*NI:(no+1)*NI], NI)
                    ptr_inter = [ArrayRef(inter.var, [mi, 0])
                                 for mi in range(MI)]
                    ptr_C = [MemRef(C.var, batch * N * L + (no * NI + ni) * L + lo)
                             for ni in range(NI)]
                    mma_m2n2xk16_acc32_aoffset(
                        ctx, ptr_inter, ptr_C, MI, NI, KI, acc2_array, pack_inter_offset)
                    for mi in range(MI):
                        for ni in range(NI):
                            D[batch * M * N + (mo * MI + mi) * N + no * NI + ni] = requantize(
                                ctx, acc2_array[mi][ni], scale_array[ni], output_offset, clip_min, clip_max
                            )


def batch_matmul_chain_s8s8s8_acc32_aoffset_golden(
        ctx, A, B, C, D, inter_scales, scales, batch, M, N, K, L, input_offset, inter_offset1, inter_offset2, output_offset, clip_min, clip_max):

    with ctx.spatial_for("batch", Range(batch)) as batch:
        with ctx.spatial_for("m", Range(M)) as m:
            inter = ctx.alloc([L], scope="local",
                              dtype="int8", name="inter")
            with ctx.spatial_for("l", Range(L)) as l:
                acc1 = ctx.alloc([1], scope="local",
                                 dtype="int32", name="acc1")
                with ctx.reduce_for("k", Range(K)) as k:
                    acc1[0] = acc1[0] + (cast("int32", A[batch * M * K + m * K + k]) +
                                         cast("int32", input_offset)) * cast("int32", B[batch * L * K + l * K + k])
                acc1[0] = cast("int32", cast("float32", acc1[0])
                               * inter_scales[batch * L + l]) + inter_offset1
                acc1[0] = clip(acc1[0], clip_min, clip_max)
                inter[l] = cast("int8", acc1[0])
            with ctx.spatial_for("n", Range(N)) as n:
                acc2 = ctx.alloc([1], scope="local",
                                 dtype="int32", name="acc2")
                with ctx.reduce_for("l", Range(L)) as l:
                    acc2[0] = acc2[0] + (cast("int32", inter[l]) +
                                         cast("int32", inter_offset2)) * cast("int32", C[batch * N * L + n * L + l])
                acc2[0] = cast("int32", cast("float32", acc2[0])
                               * scales[batch * N + n]) + output_offset
                acc2[0] = clip(acc2[0], clip_min, clip_max)
                D[batch * M * N + (m * N + n)] = cast("int8", acc2[0])


def gen_params():
    input_offset = Var("int32", "input_offset")
    inter_offset1 = Var("int32", "inter_offset1")
    inter_offset2 = Var("int32", "inter_offset2")
    output_offset = Var("int32", "output_offset")
    clip_max = Var("int32", "clip_max")
    clip_min = Var("int32", "clip_min")
    batch = Var("int32", "Batch")
    M = Var("int32", "M")
    N = Var("int32", "N")
    K = Var("int32", "K")
    L = Var("int32", "L")
    A = Tensor([batch, M, K], name="A", dtype="int8")
    B = Tensor([batch, L, K], name="B", dtype="int8")
    C = Tensor([batch, N, L], name="C", dtype="int8")
    D = Tensor([batch, M, N], name="D", dtype="int8")
    inter_scales = Tensor([batch, L], name="inter_scales", dtype="float32")
    scales = Tensor([batch, N], name="scales", dtype="float32")

    return [A, B, C, D, inter_scales, scales], [batch, M, N, K, L, input_offset, inter_offset1, inter_offset2, output_offset, clip_min, clip_max]


def gen_batch_matmul_chain():
    tensors, scalars = gen_params()
    kernel = program_build(batch_matmul_chain_s8s8s8_acc32_m2x_n4x_k16x_l16x_row_col_col_mma_m2n2k16_aoffset,
                           tensors, scalar_inputs=scalars, target="arm_m")
    return kernel, tensors, scalars


def gen_batch_matmul_chain_golden():
    tensors, scalars = gen_params()
    kernel = program_build(batch_matmul_chain_s8s8s8_acc32_aoffset_golden,
                           tensors, scalar_inputs=scalars, target="arm_m")
    return kernel, tensors, scalars
