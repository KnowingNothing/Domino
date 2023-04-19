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


def BatchGemm(ctx, tA, tB, loop_b, loop_m, loop_n, loop_k, levels):
    b, m, n = loop_b[0], loop_m[0], loop_n[0]
    tC = dir.Tensor([b.extent, m.extent, n.extent],
                    name="C", dtype="int16", ctx=ctx)

    def bmm_func(A, B, C, b, m, n, k):
        C[b, m, n] = C[b, m, n] + A[b, m, k] * B[b, k, n]

    GeneralOperator(ctx, [tA, tB, tC], [loop_b, loop_m,
                    loop_n, loop_k], levels, bmm_func)

    return tC


def BatchGemm4D(ctx, tA, tB, loop_b, loop_h, loop_m, loop_n, loop_k, levels):
    b, h, m, n = loop_b[0], loop_h[0], loop_m[0], loop_n[0]
    tC = dir.Tensor([b.extent, h.extent, m.extent, n.extent],
                    name="C", dtype="int16", ctx=ctx)

    def bmm_func(A, B, C, b, h, m, n, k):
        C[b, h, m, n] = C[b, h, m, n] + A[b, h, m, k] * B[b, h, k, n]

    GeneralOperator(ctx, [tA, tB, tC], [loop_b, loop_h, loop_m,
                    loop_n, loop_k], levels, bmm_func)

    return tC
