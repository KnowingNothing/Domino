from ..workload import GeneralOperator
import domino.program_ir as dir


def exp(ctx, tA, loop_m, loop_l, levels):
    m, l = loop_m[0], loop_l[0]
    tB = dir.Tensor([m.extent, l.extent], name="B", dtype="int16", ctx=ctx)

    def exp_func(A, B, m, l):
        B[m, l] = dir.exp(A[m, l])

    GeneralOperator(ctx, [tA, tB], [loop_m, loop_l], levels, exp_func)
    return tB
