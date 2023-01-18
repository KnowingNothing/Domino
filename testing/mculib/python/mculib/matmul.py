from domino.program_ir import *


def gen_matmul():
    MI = 2
    NI = 4
    KI = 16

    def matmul_s8s8s8_acc32_m2x_n4x_k16x_row_col_mma_m2n2k16_aoffset(
            ctx, A, B, C, scales, M, N, K, input_offset, output_offset, clip_min, clip_max):

        input_offset_s16 = ctx.map_var(
            "input_offset_s16", cast("int16", input_offset))
        pack_input_offset = ctx.map_var("pack_input_offset", pack_value(
            "int32", [input_offset_s16, input_offset_s16]))
        MO, NO, KO = [ctx.map_var(name, value) for name, value in zip(
            ["MO", "NO", "KO"], [M//MI, N//NI, K//KI])]

        with ctx.spatial_for("no", Range(NO)) as no:
            scale_array = [ctx.map_var(
                f"scale{i}", scales[no * NI + i]) for i in range(NI)]
            with ctx.spatial_for("mo", Range(MO)) as mo:
                acc_array = [[ctx.map_var(f"acc{i}{j}", make_const(
                    0, "int32")) for j in range(NI)] for i in range(MI)]
                with ctx.reduce_for("ko", Range(KO)) as ko:
                    ptr_A = [MemRef(A.var, (mo * 2 + i) * K + ko * 16)
                             for i in range(MI)]
                    ptr_B = [MemRef(B.var, ((no * 4 + i) * K + ko * 16))
                             for i in range(NI)]
                    for ni in range(2):
                        ctx.call(
                            "ignore",
                            "mma_m2n2k16_s8s8s8_acc32_aoffset",
                            (
                                ptr_A[0],
                                ptr_B[ni * NI // 2],
                                ptr_A[1],
                                ptr_B[ni * NI // 2 + 1],
                                pack_input_offset,
                                acc_array[0][ni * NI // 2],
                                acc_array[0][ni * NI // 2 + 1],
                                acc_array[1][ni * NI // 2],
                                acc_array[1][ni * NI // 2 + 1]
                            ))

                def requantize(acc_v, scale_v):
                    tmp = ctx.map_var(f"tmp_{acc_v.id.value}", cast(
                        "int32", (cast("float", acc_v) * scale_v)) + output_offset)
                    return cast("int8", clip(tmp, clip_min, clip_max))

                for mi in range(MI):
                    for ni in range(NI):
                        C[(mo * 2 + mi) * N + no * 4 +
                          ni] = requantize(acc_array[mi][ni], scale_array[ni])

    input_offset = Var("int32", "input_offset")
    output_offset = Var("int32", "output_offset")
    clip_max = Var("int32", "clip_max")
    clip_min = Var("int32", "clip_min")
    M = Var("int32", "M")
    N = Var("int32", "N")
    K = Var("int32", "K")
    A = Tensor([M, K], name="A", dtype="int8")
    B = Tensor([N, K], name="B", dtype="int8")
    C = Tensor([M, N], name="C", dtype="int8")
    scales = Tensor([N], name="scales", dtype="float32")

    kernel = program_build(matmul_s8s8s8_acc32_m2x_n4x_k16x_row_col_mma_m2n2k16_aoffset, [A, B, C, scales], scalar_inputs=[
                           M, N, K, input_offset, output_offset, clip_min, clip_max], target="arm_m")

    return kernel, [A, B, C, scales], [M, N, K, input_offset, output_offset, clip_min, clip_max]


def gen_matmul_golden():

    def matmul_s8s8s8_acc32_aoffset_golden(
            ctx, A, B, C, scales, M, N, K, input_offset, output_offset, clip_min, clip_max):

        with ctx.spatial_for("m", Range(M)) as m:
            with ctx.spatial_for("n", Range(N)) as n:
                acc = ctx.alloc([1], scope="local", dtype="int32", name="acc")
                with ctx.reduce_for("k", Range(K)) as k:
                    acc[0] = acc[0] + (cast("int32", A[m * K + k]) +
                                       cast("int32", input_offset)) * cast("int32", B[n * K + k])
                acc[0] = cast("int32", cast("float32", acc[0])
                              * scales[n]) + output_offset
                acc[0] = clip(acc[0], clip_min, clip_max)
                C[m * N + n] = cast("int8", acc[0])

    input_offset = Var("int32", "input_offset")
    output_offset = Var("int32", "output_offset")
    clip_max = Var("int32", "clip_max")
    clip_min = Var("int32", "clip_min")
    M = Var("int32", "M")
    N = Var("int32", "N")
    K = Var("int32", "K")
    A = Tensor([M, K], name="A", dtype="int8")
    B = Tensor([N, K], name="B", dtype="int8")
    C = Tensor([M, N], name="C", dtype="int8")
    scales = Tensor([N], name="scales", dtype="float32")

    kernel = program_build(matmul_s8s8s8_acc32_aoffset_golden, [A, B, C, scales], scalar_inputs=[
                           M, N, K, input_offset, output_offset, clip_min, clip_max], target="arm_m")

    return kernel, [A, B, C, scales], [M, N, K, input_offset, output_offset, clip_min, clip_max]
