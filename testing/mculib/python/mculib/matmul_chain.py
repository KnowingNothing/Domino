from domino.program_ir import *


def gen_matmul_chain():
    MI = 2
    NI = 4
    KI = 16

    def matmul_chain_s8s8s8_acc32_m2x_n4x_k16x_l16x_row_col_col_mma_m2n2k16_aoffset(
            ctx, A, B, C, D, inter_scales, scales, M, N, K, L, input_offset, inter_offset1, inter_offset2, output_offset, clip_min, clip_max):

        input_offset_s16 = ctx.map_var(
            "input_offset_s16", cast("int16", input_offset))
        pack_input_offset = ctx.map_var("pack_input_offset", pack_value(
            "int32", [input_offset_s16, input_offset_s16]))
        inter_offset_s16 = ctx.map_var(
            "inter_offset_s16", cast("int16", inter_offset2))
        pack_inter_offset = ctx.map_var("pack_inter_offset", pack_value(
            "int32", [inter_offset_s16, inter_offset_s16]))
        MO, NO, LO, KO = [ctx.map_var(name, value) for name, value in zip(
            ["MO", "NO", "LO", "KO"], [M//MI, N//NI, L//KI, K//KI])]

        def requantize(acc_v, scale_v, offset_v):
            tmp = ctx.map_var(f"tmp_{acc_v.id.value}", cast(
                "int32", (cast("float", acc_v) * scale_v)) + offset_v)
            return cast("int8", clip(tmp, clip_min, clip_max))

        with ctx.spatial_for("mo", Range(MO)) as mo:
            inter = ctx.alloc([MI, KI], scope="local",
                              dtype="int8", name="inter")
            acc2_array = [[ctx.map_var(f"acc2_{i}{j}", make_const(
                0, "int32")) for j in range(NI)] for i in range(MI)]
            with ctx.reduce_for("lo", Range(LO)) as lo:
                with ctx.spatial_for("li", Range(KI//NI)) as li:
                    inter_scale_array = [ctx.map_var(
                        f"inter_scale{ni}", inter_scales[lo * KI + li * NI + ni]) for ni in range(NI)]
                    acc1_array = [[ctx.map_var(f"acc1_{i}{j}", make_const(
                        0, "int32")) for j in range(NI)] for i in range(MI)]
                    with ctx.reduce_for("ko", Range(KO)) as ko:
                        ptr_A = [MemRef(A.var, (mo * MI + mi) * K + ko * KI)
                                 for mi in range(MI)]
                        ptr_B = [MemRef(B.var, (lo * KI + li * NI + ni) * K + ko * KI)
                                 for ni in range(NI)]
                        for ni in range(NI//2):
                            ctx.call(
                                "ignore",
                                "mma_m2n2k16_s8s8s8_acc32_aoffset",
                                (
                                    ptr_A[0],
                                    ptr_B[ni * 2],
                                    ptr_A[1],
                                    ptr_B[ni * 2 + 1],
                                    pack_input_offset,
                                    acc1_array[0][ni * 2],
                                    acc1_array[0][ni * 2 + 1],
                                    acc1_array[1][ni * 2],
                                    acc1_array[1][ni * 2 + 1]
                                )
                            )
                    for mi in range(MI):
                        for ni in range(NI):
                            inter[mi, li * NI + ni] = requantize(
                                acc1_array[mi][ni], inter_scale_array[ni], inter_offset1)

                with ctx.spatial_for("no", Range(NO)) as no:
                    scale_array = [ctx.map_var(
                        f"scale{ni}", scales[no * NI + ni]) for ni in range(NI)]
                    ptr_inter = [ArrayRef(inter.var, [mi, 0])
                                 for mi in range(MI)]
                    ptr_C = [MemRef(C.var, (no * NI + ni) * L + lo)
                             for ni in range(NI)]
                    for ni in range(NI//2):
                        ctx.call(
                            "ignore",
                            "mma_m2n2k16_s8s8s8_acc32_aoffset",
                            (
                                ptr_inter[0],
                                ptr_C[ni * 2],
                                ptr_inter[1],
                                ptr_C[ni * 2 + 1],
                                pack_inter_offset,
                                acc2_array[0][ni * 2],
                                acc2_array[0][ni * 2 + 1],
                                acc2_array[1][ni * 2],
                                acc2_array[1][ni * 2 + 1]
                            )
                        )
                    for mi in range(MI):
                        for ni in range(NI):
                            D[(mo * MI + mi) * N + no * NI + ni] = requantize(
                                acc2_array[mi][ni], scale_array[ni], output_offset
                            )

    input_offset = Var("int32", "input_offset")
    inter_offset1 = Var("int32", "inter_offset1")
    inter_offset2 = Var("int32", "inter_offset2")
    output_offset = Var("int32", "output_offset")
    clip_max = Var("int32", "clip_max")
    clip_min = Var("int32", "clip_min")
    M = Var("int32", "M")
    N = Var("int32", "N")
    K = Var("int32", "K")
    L = Var("int32", "L")
    A = Tensor([M, K], name="A", dtype="int8")
    B = Tensor([L, K], name="B", dtype="int8")
    C = Tensor([N, L], name="C", dtype="int8")
    D = Tensor([M, N], name="D", dtype="int8")
    inter_scales = Tensor([L], name="inter_scales", dtype="float32")
    scales = Tensor([N], name="scales", dtype="float32")

    kernel = program_build(matmul_chain_s8s8s8_acc32_m2x_n4x_k16x_l16x_row_col_col_mma_m2n2k16_aoffset, [A, B, C, D, inter_scales, scales], scalar_inputs=[
                           M, N, K, L, input_offset, inter_offset1, inter_offset2, output_offset, clip_min, clip_max], target="arm_m")

    return kernel, [A, B, C, D, inter_scales, scales], [M, N, K, L, input_offset, inter_offset1, inter_offset2, output_offset, clip_min, clip_max]


def gen_matmul_chain_golden():

    def matmul_chain_s8s8s8_acc32_aoffset_golden(
            ctx, A, B, C, D, inter_scales, scales, M, N, K, L, input_offset, inter_offset1, inter_offset2, output_offset, clip_min, clip_max):

        with ctx.spatial_for("m", Range(M)) as m:
            inter = ctx.alloc([L], scope="local", dtype="int8", name="inter")
            with ctx.spatial_for("l", Range(L)) as l:
                acc1 = ctx.alloc([1], scope="local",
                                 dtype="int32", name="acc1")
                with ctx.reduce_for("k", Range(K)) as k:
                    acc1[0] = acc1[0] + (cast("int32", A[m * K + k]) +
                                         cast("int32", input_offset)) * cast("int32", B[l * K + k])
                acc1[0] = cast("int32", cast("float32", acc1[0])
                               * inter_scales[l]) + inter_offset1
                acc1[0] = clip(acc1[0], clip_min, clip_max)
                inter[l] = cast("int8", acc1[0])
            with ctx.spatial_for("n", Range(N)) as n:
                acc2 = ctx.alloc([1], scope="local",
                                 dtype="int32", name="acc2")
                with ctx.reduce_for("l", Range(L)) as l:
                    acc2[0] = acc2[0] + (cast("int32", inter[l]) +
                                         cast("int32", inter_offset2)) * cast("int32", C[n * L + l])
                acc2[0] = cast("int32", cast("float32", acc2[0])
                               * scales[n]) + output_offset
                acc2[0] = clip(acc2[0], clip_min, clip_max)
                D[(m * N + n)] = cast("int8", acc2[0])

    input_offset = Var("int32", "input_offset")
    inter_offset1 = Var("int32", "inter_offset1")
    inter_offset2 = Var("int32", "inter_offset2")
    output_offset = Var("int32", "output_offset")
    clip_max = Var("int32", "clip_max")
    clip_min = Var("int32", "clip_min")
    M = Var("int32", "M")
    N = Var("int32", "N")
    K = Var("int32", "K")
    L = Var("int32", "L")
    A = Tensor([M, K], name="A", dtype="int8")
    B = Tensor([L, K], name="B", dtype="int8")
    C = Tensor([N, L], name="C", dtype="int8")
    D = Tensor([M, N], name="D", dtype="int8")
    inter_scales = Tensor([L], name="inter_scales", dtype="float32")
    scales = Tensor([N], name="scales", dtype="float32")

    kernel = program_build(matmul_chain_s8s8s8_acc32_aoffset_golden, [A, B, C, D, inter_scales, scales], scalar_inputs=[
                           M, N, K, L, input_offset, inter_offset1, inter_offset2, output_offset, clip_min, clip_max], target="arm_m")

    return kernel, [A, B, C, D, inter_scales, scales], [M, N, K, L, input_offset, inter_offset1, inter_offset2, output_offset, clip_min, clip_max]
