from domino.program_ir import *
from .quantize import requantize
from .data_process import broadcast_to_s16x2, alloc_register_array, vload_to_register_array, declare_register
from .mma import mma_m2n2xk16_acc32_aoffset


def conv2d_3x3_s8s8s8_acc32_m2x_n4x_k16x_row_col_mma_m2n2k16_aoffset(
        ctx, A, B, C, scales, N, H, W, inC, outC, input_offset, output_offset, clip_min, clip_max):
    MI = 2
    NI = 4
    KI = 16

    pack_input_offset = broadcast_to_s16x2(ctx, input_offset)
    WO, outCO, inCO = [ctx.map_var(name, value) for name, value in zip(
        ["WO", "outCO", "inCO"], [W//MI, outC//NI, inC//KI])]

    with ctx.spatial_for("batch", Range(N)) as batch:
        with ctx.spatial_for("h", Range(H)) as h:
            with ctx.spatial_for("mo", Range(WO)) as mo:
                with ctx.spatial_for("no", Range(outCO)) as no:
                    scale_array = vload_to_register_array(
                        ctx, "scale", scales[no*NI:(no+1)*NI], NI)
                    acc_array = alloc_register_array(
                        ctx, [MI, NI], "int32", "acc", 0)
                    with ctx.reduce_for("r", Range(3)) as r:
                        with ctx.reduce_for("s", Range(3)) as s:
                            with ctx.reduce_for("ko", Range(inCO)) as ko:
                                ptr_A = [A.ref(batch, (h + r), (mo * 2 + i) + s, ko * 16)
                                         for i in range(MI)]
                                ptr_B = [B.ref(r, s, no * 4 + i, ko * 16)
                                         for i in range(NI)]
                                mma_m2n2xk16_acc32_aoffset(
                                    ctx, ptr_A, ptr_B, MI, NI, KI, acc_array, pack_input_offset)
                            for mi in range(MI):
                                for ni in range(NI):
                                    C[batch, h, (mo * 2 + mi), no * 4 + ni] = requantize(ctx, acc_array[mi]
                                                                                         [ni], scale_array[ni], output_offset, clip_min, clip_max)


def conv2d_3x3_s8s8s8_acc32_m2x_n4x_k16x_row_col_mma_m2n2k16_aoffset_ring_buffer(
        ctx, A, B, C, scales, N, H, W, inC, outC, input_offset, output_offset, clip_min, clip_max):
    MI = 2
    NI = 4
    KI = 16

    pack_input_offset = broadcast_to_s16x2(ctx, input_offset)
    WO, outCO, inCO = [ctx.map_var(name, value) for name, value in zip(
        ["WO", "outCO", "inCO"], [W//MI, outC//NI, inC//KI])]

    with ctx.spatial_for("batch", Range(N)) as batch:
        with ctx.spatial_for("h", Range(H)) as h:
            with ctx.spatial_for("mo", Range(WO)) as mo:
                ctx.attr("ring_buffer_check_bound",
                         C.var, C[batch, h, mo * 2, 0])
                with ctx.spatial_for("no", Range(outCO)) as no:
                    scale_array = vload_to_register_array(
                        ctx, "scale", scales[no*NI:(no+1)*NI], NI)
                    acc_array = alloc_register_array(
                        ctx, [MI, NI], "int32", "acc", 0)
                    with ctx.reduce_for("r", Range(3)) as r:
                        with ctx.reduce_for("s", Range(3)) as s:
                            with ctx.reduce_for("ko", Range(inCO)) as ko:
                                ptr_A = [A.ref(batch, (h + r), (mo * 2 + i) + s, ko * 16)
                                         for i in range(MI)]
                                ptr_B = [B.ref(r, s, no * 4 + i, ko * 16)
                                         for i in range(NI)]
                                mma_m2n2xk16_acc32_aoffset(
                                    ctx, ptr_A, ptr_B, MI, NI, KI, acc_array, pack_input_offset)
                            for mi in range(MI):
                                for ni in range(NI):
                                    C[batch, h, (mo * 2 + mi), no * 4 + ni] = requantize(ctx, acc_array[mi]
                                                                                         [ni], scale_array[ni], output_offset, clip_min, clip_max)


def conv2d_3x3_s8s8s8_acc32_aoffset_golden(
        ctx, A, B, C, scales, N, H, W, inC, outC, input_offset, output_offset, clip_min, clip_max):

    with ctx.spatial_for("batch", Range(N)) as batch:
        with ctx.spatial_for("h", Range(H)) as h:
            with ctx.spatial_for("w", Range(W)) as w:
                with ctx.spatial_for("k", Range(outC)) as k:
                    acc = ctx.alloc([1], scope="local",
                                    dtype="int32", name="acc")
                    with ctx.reduce_for("r", Range(3)) as r:
                        with ctx.reduce_for("s", Range(3)) as s:
                            with ctx.reduce_for("c", Range(inC)) as c:
                                acc[0] = acc[0] + (cast("int32", A[batch, h+r, w+s, c]) +
                                                   cast("int32", input_offset)) * cast("int32", B[r, s, k, c])
                    acc[0] = cast("int32", cast("float32", acc[0])
                                  * scales[k]) + output_offset
                    acc[0] = clip(acc[0], clip_min, clip_max)
                    C[batch, h, w, k] = cast("int8", acc[0])


def gen_params():
    input_offset = Var("int32", "input_offset")
    output_offset = Var("int32", "output_offset")
    clip_max = Var("int32", "clip_max")
    clip_min = Var("int32", "clip_min")
    N = Var("int32", "N")
    H = Var("int32", "H")
    W = Var("int32", "W")
    inC = Var("int32", "inC")
    outC = Var("int32", "outC")
    A = Tensor([N, H + 2, W + 2, inC], name="A", dtype="int8")
    B = ConstTensor([3, 3, outC, inC], name="B", dtype="int8")
    C = Tensor([N, H, W, outC], name="C", dtype="int8")
    scales = ConstTensor([N], name="scales", dtype="float32")
    return [A, B, C, scales], [N, H, W, inC, outC, input_offset, output_offset, clip_min, clip_max]


def gen_conv2d():
    tensors, scalars = gen_params()
    kernel = program_build(conv2d_3x3_s8s8s8_acc32_m2x_n4x_k16x_row_col_mma_m2n2k16_aoffset,
                           tensors, scalar_inputs=scalars, target="arm_m")
    return kernel, tensors, scalars


def gen_conv2d_ring_buffer():
    tensors, scalars = gen_params()
    # print_ir(program_lower(matmul_s8s8s8_acc32_m2x_n4x_k16x_row_col_mma_m2n2k16_aoffset_ring_buffer,
    #                       tensors, scalar_inputs=scalars))
    kernel = program_build(conv2d_3x3_s8s8s8_acc32_m2x_n4x_k16x_row_col_mma_m2n2k16_aoffset_ring_buffer,
                           tensors, scalar_inputs=scalars, target="arm_m")
    return kernel, tensors, scalars


def gen_conv2d_golden():
    tensors, scalars = gen_params()

    kernel = program_build(conv2d_3x3_s8s8s8_acc32_aoffset_golden,
                           tensors, scalar_inputs=scalars, target="arm_m")

    return kernel, tensors, scalars
