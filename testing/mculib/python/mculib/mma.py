from domino.program_ir import *


def mma_m2n2xk16_acc32_aoffset(ctx, ptr_A, ptr_B, MI, NI, KI, acc_array, pack_input_offset):
    """
    acc_array: [MI, NI]
    A: [M, K]
    B: [N, K]
    MI = 2
    NI = 2x
    KI = 16
    """
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
                acc_array[0][ni * 2],
                acc_array[0][ni * 2 + 1],
                acc_array[1][ni * 2],
                acc_array[1][ni * 2 + 1]
            )
        )
