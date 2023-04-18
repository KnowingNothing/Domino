from ..workload import GeneralOperator
import domino.program_ir as dir


def exp(ctx, tA, loop_m, loop_l, levels):
    m, l = loop_m[0], loop_l[0]
    tB = dir.Tensor([m.extent, l.extent], name="B", dtype="int16", ctx=ctx)

    def exp_func(A, B, m, l):
        B[m, l] = dir.exp(A[m, l])

    GeneralOperator(ctx, [tA, tB], [loop_m, loop_l], levels, exp_func)
    return tB


def exp3D(ctx, tA, loop_b, loop_m, loop_l, levels):
    b, m, l = loop_b[0], loop_m[0], loop_l[0]
    tB = dir.Tensor([b.extent, m.extent, l.extent],
                    name="B", dtype="int16", ctx=ctx)

    def exp_func(A, B, b, m, l):
        B[b, m, l] = dir.exp(A[b, m, l])

    GeneralOperator(ctx, [tA, tB], [loop_b, loop_m, loop_l], levels, exp_func)
    return tB


def sum3D(ctx, tA, loop_b, loop_m, loop_n, levels):
    b, m = loop_b[0], loop_m[0]
    tB = dir.Tensor([b.extent, m.extent], name="B", dtype="int16", ctx=ctx)

    def sum_func(A, B, b, m, n):
        B[b, m] = B[b, m] + A[b, m, n]

    GeneralOperator(ctx, [tA, tB], [loop_b, loop_m, loop_n], levels, sum_func)

    return tB


def max3D(ctx, tA, loop_b, loop_m, loop_n, levels):
    b, m = loop_b[0], loop_m[0]
    tB = dir.Tensor([b.extent, m.extent], name="B", dtype="int16", ctx=ctx)

    def max_func(A, B, b, m, n):
        B[b, m] = dir.max(B[b, m], A[b, m, n])

    GeneralOperator(ctx, [tA, tB], [loop_b, loop_m, loop_n], levels, max_func)

    return tB


def sub3D(ctx, tA, tB, loop_b, loop_m, loop_l, levels):
    b, m, l = loop_b[0], loop_m[0], loop_l[0]
    tC = dir.Tensor([b.extent, m.extent, l.extent],
                    name="C", dtype="int16", ctx=ctx)

    def sub_func(A, B, C, b, m, l):
        C[b, m, l] = A[b, m, l] - B[b, m, l]

    GeneralOperator(ctx, [tA, tB, tC], [
                    loop_b, loop_m, loop_l], levels, sub_func)
    return tC


def div3D(ctx, tA, tB, loop_b, loop_m, loop_l, levels):
    b, m, l = loop_b[0], loop_m[0], loop_l[0]
    tC = dir.Tensor([b.extent, m.extent, l.extent],
                    name="C", dtype="int16", ctx=ctx)

    def div_func(A, B, C, b, m, l):
        C[b, m, l] = A[b, m, l] / B[b, m, l]

    GeneralOperator(ctx, [tA, tB, tC], [
                    loop_b, loop_m, loop_l], levels, div_func)
    return tC


def softmax3D(ctx, tA, loop_b, loop_m, loop_l, levels):
    tB = max3D(ctx, tA, loop_b, loop_m, loop_l, levels)
    tC = sub3D(ctx, tA, tB, loop_b, loop_m, loop_l, levels)
    tD = exp3D(ctx, tC, loop_b, loop_m, loop_l, levels)
    tE = sum3D(ctx, tD, loop_b, loop_m, loop_l, levels)
    tF = div3D(ctx, tB, tE, loop_b, loop_m, loop_l, levels)
    return tF
