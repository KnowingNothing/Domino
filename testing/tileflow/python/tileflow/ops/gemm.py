from ..workload import GeneralOperator
import domino.program_ir as dir


def Gemm(ctx, tA, tB, loop_m, loop_l, loop_k, levels):
    m, l = loop_m[0], loop_l[0]
    tC = dir.Tensor([m.extent, l.extent], name="C", dtype="int16", ctx=ctx)

    def gemm_func(A, B, C, m, l, k):
        C[m, l] = C[m, l] + A[m, k] * B[k, l]

    GeneralOperator(ctx, [tA, tB, tC], [
                    loop_m, loop_l, loop_k], levels, gemm_func)
    return tC
