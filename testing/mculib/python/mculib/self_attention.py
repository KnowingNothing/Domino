from domino.program_ir import *
from .quantize import requantize
from .data_process import broadcast_to_s16x2, alloc_register_array, vload_to_register_array
from .mma import mma_m2n2xk16_acc32_aoffset
import math


def self_attention_s8s8s8_acc32_seq16x_hidden16xheads_mma_m2n2k16_aoffset(
        ctx, A, W0, W1, W2, Output, matmul0_scales, matmul1_scales, matmul2_scales, bmm0_scales, softmax_exp_scales, softmax_div_scales, bmm1_scales, seq_len, hidden, num_heads, input_offset, matmul0_offset, matmul1_offset, matmul2_offset, bmm0_input_offset, bmm0_output_offset, softmax_input_offset, softmax_exp_offset, softmax_output_offset, bmm1_input_offset, bmm1_output_offset, clip_min, clip_max):
    # A: [seq_len, hidden]
    # W0: [hidden, hidden]
    # W1: [hidden, hidden]
    # W2: [hidden, hidden]
    MI = 2
    NI = 4
    KI = 16
    # model_k should be multiple of NI
    model_k = hidden // num_heads

    pack_input_offset = broadcast_to_s16x2(ctx, input_offset)
    pack_bmm0_input_offset = broadcast_to_s16x2(ctx, bmm0_input_offset)
    pack_bmm1_input_offset = broadcast_to_s16x2(ctx, bmm1_input_offset)

    D0 = ctx.alloc([MI, model_k], scope="local", dtype="int8", name="D0")
    D1 = ctx.alloc([seq_len, model_k], scope="local", dtype="int8", name="D1")
    D2 = ctx.alloc([model_k, seq_len], scope="local", dtype="int8", name="D2")
    E = ctx.alloc([MI, seq_len], scope="local", dtype="int8", name="E")

    # one head at a time
    with ctx.spatial_for("h", Range(num_heads)) as h:
        # compute K, V
        with ctx.spatial_for("io", Range(seq_len//MI)) as io:
            with ctx.spatial_for("jo", Range(model_k//NI)) as jo:
                acc1_array = alloc_register_array(
                    ctx, [MI, NI], "int32", "acc1", 0)
                acc2_array = alloc_register_array(
                    ctx, [MI, NI], "int32", "acc2", 0)
                matmul1_scale_array = vload_to_register_array(
                    ctx, "matmul1_scale_array", matmul1_scales.ref(h * model_k + jo * NI), NI)
                matmul2_scale_array = vload_to_register_array(
                    ctx, "matmul2_scale_array", matmul2_scales.ref(h * model_k + jo * NI), NI)
                with ctx.reduce_for("ko", Range(hidden//KI)) as ko:
                    ptr_A = [A.ref(io * MI + ii, ko * KI) for ii in range(MI)]
                    ptr_W1 = [W1.ref(h * model_k + jo * NI + ji, ko * KI)
                              for ji in range(NI)]
                    ptr_W2 = [W2.ref(h * model_k + jo * NI + ji, ko * KI)
                              for ji in range(NI)]
                    mma_m2n2xk16_acc32_aoffset(
                        ctx, ptr_A, ptr_W1, MI, NI, KI, acc1_array, pack_input_offset
                    )
                    mma_m2n2xk16_acc32_aoffset(
                        ctx, ptr_A, ptr_W2, MI, NI, KI, acc2_array, pack_input_offset
                    )
                for ii in range(MI):
                    for ji in range(NI):
                        D1[io * MI + ii, jo * NI + ji] = requantize(
                            ctx, acc1_array[ii][ji], matmul1_scale_array[ji], matmul1_offset, clip_min, clip_max)
                        D2[jo * NI + ji, io * MI + ii] = requantize(
                            ctx, acc2_array[ii][ji], matmul2_scale_array[ji], matmul2_offset, clip_min, clip_max)

        with ctx.spatial_for("io", Range(seq_len//MI)) as io:
            # compute part of Q
            with ctx.spatial_for("jo", Range(model_k//NI)) as jo:
                acc0_array = alloc_register_array(
                    ctx, [MI, NI], "int32", "acc0", 0)
                matmul0_scale_array = vload_to_register_array(
                    ctx, "matmul0_scale_array", matmul0_scales.ref(h * model_k + jo * NI), NI)
                with ctx.reduce_for("ko", Range(hidden//KI)) as ko:
                    ptr_A = [A.ref(io * MI + ii, ko * KI) for ii in range(MI)]
                    ptr_W0 = [W0.ref(h * model_k + jo * NI + ji, ko * KI)
                              for ji in range(NI)]
                    mma_m2n2xk16_acc32_aoffset(
                        ctx, ptr_A, ptr_W0, MI, NI, KI, acc0_array, pack_input_offset
                    )
                for ii in range(MI):
                    for ji in range(NI):
                        D0[ii, jo * NI + ji] = requantize(
                            ctx, acc0_array[ii][ji], matmul0_scale_array[ji], matmul0_offset, clip_min, clip_max
                        )
            # compute part of Q * K
            with ctx.spatial_for("jo", Range(seq_len//NI)) as jo:
                bmm0_acc_array = alloc_register_array(
                    ctx, [MI, NI], "int32", "bmm0_acc", 0)
                bmm0_scale_array = vload_to_register_array(
                    ctx, "bmm0_scale_array", bmm0_scales.ref(h, jo * NI), NI)
                with ctx.reduce_for("ko", Range(model_k//KI)) as ko:
                    ptr_D0 = [D0.ref(ii, ko * KI) for ii in range(MI)]
                    ptr_D1 = [D1.ref(jo * NI + ji, ko * KI)
                              for ji in range(NI)]
                    mma_m2n2xk16_acc32_aoffset(
                        ctx, ptr_D0, ptr_D1, MI, NI, KI, bmm0_acc_array, pack_bmm0_input_offset
                    )
                for ii in range(MI):
                    for ji in range(NI):
                        E[ii, jo * NI + ji] = requantize(
                            ctx, bmm0_acc_array[ii][ji], bmm0_scale_array[ji], bmm0_output_offset, clip_min, clip_max
                        )

            # compute part of softmax
            # TODO: we may need to use the I-Bert method: https://arxiv.org/pdf/2101.01321.pdf
            a, b, c = [0.5, 1.0, 1.0]
            a, b, c = make_const(a, "float32"), make_const(
                b, "float32"), make_const(c, "float32")
            # ln2 = make_const(-math.log(2), "float32")
            sumv = ctx.alloc([MI], scope="local", dtype="int32", name="sumv")
            maxv = ctx.alloc([MI], scope="local", dtype="int8", name="maxv")
            # find max value
            for ii in range(MI):
                sumv[ii] = make_const(0, "int32")
                maxv[ii] = cast("int8", clip_min)
            with ctx.spatial_for("jo", Range(seq_len//NI)) as jo:
                for ii in range(MI):
                    for ji in range(NI):
                        maxv[ii] = Max(maxv[ii], E[ii, jo * NI + ji])
            with ctx.spatial_for("jo", Range(seq_len//NI)) as jo:
                for ii in range(MI):
                    for ji in range(NI):
                        # all the results are negative int8

                        q = E[ii, jo * NI + ji] - maxv[ii]
                        q = cast("float32", q) + \
                            cast("float32", softmax_input_offset)
                        # # the result is positive int32
                        # z = cast("int32", q / ln2)
                        # # get the remainder, negative (-ln2, 0]
                        # q = q - cast("float32", z) * ln2
                        # # polynomial approximation in (-ln2, 0]
                        # expv = a * (q * (q + b) + c)
                        # # restore the real exp
                        # expv = expv / cast("float32", (make_const(1, "int32") << z))

                        expv = exp(q)
                        # expv = a * q * q + b * q + c
                        E[ii, jo * NI + ji] = requantize(
                            ctx, expv, softmax_exp_scales[h, jo * NI + ji], softmax_exp_offset, clip_min, clip_max)
                        sumv[ii] = sumv[ii] + \
                            cast("int32", E[ii, jo * NI + ji])
            with ctx.spatial_for("jo", Range(seq_len//NI)) as jo:
                for ii in range(MI):
                    for ji in range(NI):
                        v = cast("float32", E[ii, jo * NI + ji]
                                 ) / cast("float32", sumv[ii])
                        E[ii, jo * NI + ji] = requantize(
                            ctx, v, softmax_div_scales[h, jo * NI + ji], softmax_output_offset, clip_min, clip_max)

            # compute part of Q * K * V
            with ctx.spatial_for("jo", Range(model_k//NI)):
                bmm1_acc_array = alloc_register_array(
                    ctx, [MI, NI], "int32", "bmm1_acc", 0
                )
                bmm1_scale_array = vload_to_register_array(
                    ctx, "bmm1_scale_array", bmm1_scales.ref(
                        h * model_k + jo * NI), NI
                )
                with ctx.reduce_for("ko", Range(seq_len//KI)):
                    ptr_E = [E.ref(ii, ko * KI) for ii in range(MI)]
                    ptr_D2 = [D2.ref(jo * NI + ji, ko * KI)
                              for ji in range(NI)]
                    mma_m2n2xk16_acc32_aoffset(
                        ctx, ptr_E, ptr_D2, MI, NI, KI, bmm1_acc_array, pack_bmm1_input_offset
                    )
                for ii in range(MI):
                    for ji in range(NI):
                        Output[io * MI + ii, h * model_k + jo * NI + ji] = requantize(
                            ctx, bmm1_acc_array[ii][ji], bmm1_scale_array[ji], bmm1_output_offset, clip_min, clip_max
                        )


def gen_params():
    seq_len = Var("int32", "SeqLen")
    hidden = Var("int32", "Hidden")
    num_heads = Var("int32", "NumHeads")
    input_offset = Var("int32", "input_offset")
    matmul0_offset = Var("int32", "matmul0_offset")
    matmul1_offset = Var("int32", "matmul1_offset")
    matmul2_offset = Var("int32", "matmul2_offset")
    bmm0_input_offset = Var("int32", "bmm0_input_offset")
    bmm0_output_offset = Var("int32", "bmm0_output_offset")
    bmm1_input_offset = Var("int32", "bmm1_input_offset")
    bmm1_output_offset = Var("int32", "bmm1_output_offset")
    softmax_input_offset = Var("int32", "softmax_input_offset")
    softmax_exp_offset = Var("int32", "softmax_exp_offset")
    softmax_output_offset = Var("int32", "softmax_output_offset")
    clip_max = Var("int32", "clip_max")
    clip_min = Var("int32", "clip_min")

    A = Tensor([seq_len, hidden], name="A", dtype="int8")
    Output = Tensor([seq_len, hidden], name="Output", dtype="int8")
    W0 = ConstTensor([hidden, hidden], name="W0", dtype="int8")
    W1 = ConstTensor([hidden, hidden], name="W1", dtype="int8")
    W2 = ConstTensor([hidden, hidden], name="W2", dtype="int8")
    matmul0_scales = ConstTensor(
        [hidden], name="matmul0_scales", dtype="float32")
    matmul1_scales = ConstTensor(
        [hidden], name="matmul1_scales", dtype="float32")
    matmul2_scales = ConstTensor(
        [hidden], name="matmul2_scales", dtype="float32")
    bmm0_scales = ConstTensor([num_heads, seq_len],
                              name="bmm0_scales", dtype="float32")
    bmm1_scales = ConstTensor([hidden], name="bmm1_scales", dtype="float32")
    softmax_exp_scales = ConstTensor(
        [num_heads, seq_len], name="softmax_exp_scales", dtype="float32")
    softmax_div_scales = ConstTensor(
        [num_heads, seq_len], name="softmax_div_scales", dtype="float32")

    tensors = [A, W0, W1, W2, Output, matmul0_scales, matmul1_scales, matmul2_scales,
               bmm0_scales, softmax_exp_scales, softmax_div_scales, bmm1_scales]
    scalars = [seq_len, hidden, num_heads, input_offset, matmul0_offset, matmul1_offset,
               matmul2_offset, bmm0_input_offset, bmm0_output_offset, softmax_input_offset,
               softmax_exp_offset, softmax_output_offset, bmm1_input_offset, bmm1_output_offset,
               clip_min, clip_max]

    return tensors, scalars


def self_attention_s8s8s8_acc32_golden(ctx, A, W0, W1, W2, Output, matmul0_scales, matmul1_scales, matmul2_scales, bmm0_scales, softmax_exp_scales, softmax_div_scales, bmm1_scales, seq_len, hidden, num_heads, input_offset, matmul0_offset, matmul1_offset, matmul2_offset, bmm0_input_offset, bmm0_output_offset, softmax_input_offset, softmax_exp_offset, softmax_output_offset, bmm1_input_offset, bmm1_output_offset, clip_min, clip_max):
    Q = ctx.alloc([seq_len, hidden], scope="local", dtype="int8", name="Q")
    K = ctx.alloc([seq_len, hidden], scope="local", dtype="int8", name="K")
    V = ctx.alloc([seq_len, hidden], scope="local", dtype="int8", name="V")
    QK = ctx.alloc([num_heads, seq_len, seq_len],
                   scope="local", dtype="int8", name="QK")

    model_k = hidden // num_heads
    with ctx.spatial_for("i", Range(seq_len)) as i:
        with ctx.spatial_for("j", Range(hidden)) as j:
            acc0 = ctx.alloc([1], scope="local", dtype="int32", name="acc0")
            acc1 = ctx.alloc([1], scope="local", dtype="int32", name="acc1")
            acc2 = ctx.alloc([1], scope="local", dtype="int32", name="acc2")
            acc0[0] = make_const(0, "int32")
            acc1[0] = make_const(0, "int32")
            acc2[0] = make_const(0, "int32")
            with ctx.reduce_for("k", Range(hidden)) as k:
                acc0[0] = acc0[0] + (cast("int32", A[i, k]) +
                                     cast("int32", input_offset)) * cast("int32", W0[j, k])
                acc1[0] = acc1[0] + (cast("int32", A[i, k]) +
                                     cast("int32", input_offset)) * cast("int32", W1[j, k])
                acc2[0] = acc2[0] + (cast("int32", A[i, k]) +
                                     cast("int32", input_offset)) * cast("int32", W2[j, k])
            Q[i, j] = cast("int8", clip(cast("int32", cast(
                "float32", acc0[0]) * matmul0_scales[j]) + matmul0_offset, clip_min, clip_max))
            K[i, j] = cast("int8", clip(cast("int32", cast(
                "float32", acc1[0]) * matmul1_scales[j]) + matmul1_offset, clip_min, clip_max))
            V[i, j] = cast("int8", clip(cast("int32", cast(
                "float32", acc2[0]) * matmul2_scales[j]) + matmul2_offset, clip_min, clip_max))
    with ctx.spatial_for("h", Range(num_heads)) as h:
        with ctx.spatial_for("i", Range(seq_len)) as i:
            with ctx.spatial_for("j", Range(seq_len)) as j:
                qk = ctx.alloc([1], scope="local", dtype="int32", name="qk")
                qk[0] = make_const(0, "int32")
                with ctx.reduce_for("k", Range(model_k)) as k:
                    qk[0] = qk[0] + (cast("int32", (Q[i, h * model_k + k])) + cast(
                        "int32", bmm0_input_offset)) * cast("int32", K[j, h * model_k + k])
                QK[h, i, j] = cast("int8", clip(cast("int32", cast(
                    "float32", qk[0]) * bmm0_scales[h, j]) + bmm0_output_offset, clip_min, clip_max))
    with ctx.spatial_for("h", Range(num_heads)) as h:
        with ctx.spatial_for("i", Range(seq_len)) as i:
            maxv = ctx.alloc([1], scope="local", dtype="int8", name="maxv")
            sumv = ctx.alloc([1], scope="local", dtype="int32", name="sumv")
            expv = ctx.alloc([seq_len], scope="local",
                             dtype="float32", name="expv")
            maxv[0] = cast("int8", clip_min)
            sumv[0] = make_const(0, "int32")
            with ctx.reduce_for("j", Range(seq_len)) as j:
                maxv[0] = Max(maxv[0], QK[h, i, j])
            with ctx.spatial_for("j", Range(seq_len)) as j:
                QK[h, i, j] = QK[h, i, j] - maxv[0]
            # a, b, c = [0.35815147, 0.96963238, 1.0]
            # b = b / a
            # c = c / a
            # a, b, c = make_const(a, "float32"), make_const(
            #     b, "float32"), make_const(c, "float32")
            # ln2 = make_const(-math.log(2), "float32")
            with ctx.spatial_for("j", Range(seq_len)) as j:
                q = (cast("float32", QK[h, i, j]) +
                     cast("float32", softmax_input_offset))

                # # the result is positive int32
                # z = cast("int32", q / ln2)
                # # get the remainder, negative (-ln2, 0]
                # q = q - cast("float32", z) * ln2
                # # polynomial approximation in (-ln2, 0]
                # p = a * (q * (q + b) + c)
                # # restore the real exp
                # expv[j] = p / cast("float32", (make_const(1, "int32") << z))

                expv[j] = exp(q)
            with ctx.spatial_for("j", Range(seq_len)) as j:
                QK[h, i, j] = requantize(
                    ctx, expv[j], softmax_exp_scales[h, j], softmax_exp_offset, clip_min, clip_max)
            with ctx.reduce_for("j", Range(seq_len)) as j:
                sumv[0] = sumv[0] + cast("int32", QK[h, i, j])
            with ctx.spatial_for("j", Range(seq_len)) as j:
                v = cast("float32", QK[h, i, j]) / cast("float32", sumv[0])
                QK[h, i, j] = requantize(
                    ctx, v, softmax_div_scales[h, j], softmax_output_offset, clip_min, clip_max)

    with ctx.spatial_for("h", Range(num_heads)) as h:
        with ctx.spatial_for("i", Range(seq_len)) as i:
            with ctx.spatial_for("j", Range(model_k)) as j:
                qkv = ctx.alloc([1], scope="local", dtype="int32", name="qkv")
                qkv[0] = make_const(0, "int32")
                with ctx.reduce_for("k", Range(seq_len)) as k:
                    qkv[0] = qkv[0] + (cast("int32", (QK[h, i, k])) + cast(
                        "int32", bmm1_input_offset)) * cast("int32", V[k, h * model_k + j])
                Output[i, h * model_k + j] = cast("int8", clip(cast("int32", cast(
                    "float32", qkv[0]) * bmm1_scales[h * model_k + j]) + bmm1_output_offset, clip_min, clip_max))


def gen_self_attention():
    tensors, scalars = gen_params()
    kernel = program_build(self_attention_s8s8s8_acc32_seq16x_hidden16xheads_mma_m2n2k16_aoffset,
                           tensors, scalar_inputs=scalars, target="arm_m")
    return kernel, tensors, scalars


def gen_self_attention_golden():
    tensors, scalars = gen_params()
    kernel = program_build(self_attention_s8s8s8_acc32_golden,
                           tensors, scalar_inputs=scalars, target="arm_m")
    return kernel, tensors, scalars
