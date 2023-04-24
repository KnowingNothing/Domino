from ..workload import GeneralOperator
import domino.program_ir as dir


def Conv2d(ctx, tA, tB, loop_b, loop_p, loop_q, loop_c, loop_k, loop_r, loop_s, levels):
    b, p, q, c, k, r, s = loop_b[0], loop_p[0], loop_q[0], loop_c[0], loop_k[0], loop_r[0], loop_s[0]
    tC = dir.Tensor([b.extent, p.extent, q.extent, k.extent],
                    name="C", dtype="int16", ctx=ctx)

    def conv_func(A, B, C, b, p, q, c, k, r, s):
        C[b, p, q, k] = C[b, p, q, k] + A[b, p + r, q + s, c] * B[k, r, s, c]

    GeneralOperator(ctx, [tA, tB, tC], [
                    loop_b, loop_p, loop_q, loop_c, loop_k, loop_r, loop_s], levels, conv_func)
    return tC


def Conv2d_nchw(ctx, tA, tB, loop_b, loop_p, loop_q, loop_c, loop_k, loop_r, loop_s, levels):
    b, p, q, c, k, r, s = loop_b[0], loop_p[0], loop_q[0], loop_c[0], loop_k[0], loop_r[0], loop_s[0]
    tC = dir.Tensor([b.extent, p.extent, q.extent, k.extent],
                    name="C", dtype="int16", ctx=ctx)

    def conv_func(A, B, C, b, p, q, c, k, r, s):
        C[b, k, p, q] = C[b, k, p, q] + A[b, c, p + r, q + s] * B[k, c, r, s]

    GeneralOperator(ctx, [tA, tB, tC], [
                    loop_b, loop_p, loop_q, loop_c, loop_k, loop_r, loop_s], levels, conv_func)
    return tC