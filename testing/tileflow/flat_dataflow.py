from domino.program_ir import *
from domino.analysis import *
import numpy as np


def self_attention(batch, num_heads, seq_len, model_k):
    def bmm1(tA, tB):
        b, h, m, n = [SLoop(x, name=y) for x, y in zip([batch, num_heads, seq_len, seq_len], ["b", "h", "m", "n"])]
        k = RLoop(model_k, name="k")
        tS = Tensor([batch, num_heads, seq_len, seq_len], dtype="int8", name="tL")
        tS.Init([b, h, m, n], const(0, "int8"))
        tS.Update([b, h, m, n], tA[b, h, m, k] * tB[b, h, n, k], op=ReduceOp("sum"), reduce_axis=[k])
        return tS
    
    def bmm2(tA, tB):
        b, h, m, n = [SLoop(x, name=y) for x, y in zip([batch, num_heads, seq_len, model_k], ["b", "h", "m", "n"])]
        k = RLoop(seq_len, name="k")
        tS = Tensor([batch, num_heads, seq_len, model_k], dtype="int8", name="tA")
        tS.Init([b, h, m, n], const(0, "int8"))
        tS.Update([b, h, m, n], tA[b, h, m, k] * tB[b, h, n, k], op=ReduceOp("sum"), reduce_axis=[k])
        return tS

    def maxv(tA):
        b, h, m = [SLoop(x, name=y) for x, y in zip([batch, num_heads, seq_len], ["b", "h", "m"])]
        k = RLoop(seq_len, name="k")
        tB = Tensor([batch, num_heads, seq_len], dtype="int8", name="tMaxv")
        tB.Init([b, h, m], const(-128, "int8"))
        tB.Update([b, h, m], tA[b, h, m, k], op=ReduceOp("max"), reduce_axis=[k])
        return tB

    def sub(tA, tB):
        b, h, m, n = [SLoop(x, name=y) for x, y in zip([batch, num_heads, seq_len, seq_len], ["b", "h", "m", "n"])]
        tC = Tensor([batch, num_heads, seq_len, seq_len], dtype="int8", name="tSub")
        tC[b, h, m, n] = tA[b, h, m, n] - tB[b, h, m]
        return tC

    def exp(tA):
        b, h, m, n = [SLoop(x, name=y) for x, y in zip([batch, num_heads, seq_len, seq_len], ["b", "h", "m", "n"])]
        tB = Tensor([batch, num_heads, seq_len, seq_len], dtype="int8", name="tExp")
        tB[b, h, m, n] = Call("int8", "exp", [tA[b, h, m, n].as_expr()])
        return tB

    def sumv(tA):
        b, h, m = [SLoop(x, name=y) for x, y in zip([batch, num_heads, seq_len], ["b", "h", "m"])]
        k = RLoop(seq_len, name="k")
        tB = Tensor([batch, num_heads, seq_len], dtype="int8", name="tSumv")
        tB.Init([b, h, m], const(0, "int8"))
        tB.Update([b, h, m], tA[b, h, m, k], op=ReduceOp("sum"), reduce_axis=[k])
        return tB

    def div(tA, tB):
        b, h, m, n = [SLoop(x, name=y) for x, y in zip([batch, num_heads, seq_len, seq_len], ["b", "h", "m", "n"])]
        tC = Tensor([batch, num_heads, seq_len, seq_len], dtype="int8", name="tDiv")
        tC[b, h, m, n] = tA[b, h, m, n] / tB[b, h, m]
        return tC

    def softmax(tA):
        T1 = maxv(tA)
        T2 = sub(tA, T1)
        T3 = exp(T2)
        T4 = sumv(T3)
        T5 = div(T3, T4)
        return T5

    Q = Tensor([batch, num_heads, seq_len, model_k], name="tQ", dtype="int8")
    K = Tensor([batch, num_heads, seq_len, model_k], name="tK", dtype="int8")
    V = Tensor([batch, num_heads, model_k, seq_len], name="tV", dtype="int8")

    L = bmm1(Q, K)
    S = softmax(L)
    A = bmm2(S, V)
    return A


if __name__ == "__main__":
    batch_size = 64
    num_heads = 12
    seq_len = 1024
    hidden_size = 768
    model_k = hidden_size//num_heads
    T = self_attention(batch_size, num_heads, seq_len, model_k)
    dag = make_prod_consum_graph(T)
    tQ, tK, tBmm1, tMax, tSub, tExp, tSum, tDiv, tV, tBmm2 = dag.nodes
    mem_levels = [0, 1, 2]
    trees = {
        t: create_tree(t, mem_levels) for t in dag.nodes
    }

    print("########################################")
    print("Initial Memory Level Trees:")
    for k, v in trees.items():
        print_ir(v.root)

    print("########################################")
    print("Create FLAT Dataflow:")
    root_tree = trees[tBmm2]
    # bmm2 factors
    bmm2_batch_f0 = 1
    bmm2_batch_f1 = 1
    bmm2_head_f0 = 1
    bmm2_head_f1 = 1
    bmm2_row_f0 = 4
    bmm2_row_f1 = 1
    bmm2_col_f0 = 16
    bmm2_col_f1 = 1
    bmm2_reduce_f0 = 16
    bmm2_reduce_f1 = 1
    # from consumer to producer
    for t in reversed([tBmm1, tMax, tSub, tExp, tSum, tDiv]):
        t_tree = trees[t]
        root_tree = root_tree.merge(t_tree, tBmm2.var, 2)
    # tile tBmm2
    b, h, m, n, k = tBmm2.updates[0].all_loops()
    tBmm2_tile = {
        b: [bmm2_batch_f0, bmm2_batch_f1, batch_size//(bmm2_batch_f0*bmm2_batch_f1)],
        h: [bmm2_head_f0, bmm2_head_f1, num_heads//(bmm2_head_f0 * bmm2_head_f1)],
        m: [bmm2_row_f0, bmm2_row_f1, seq_len//(bmm2_row_f0 * bmm2_row_f1)],
        n: [bmm2_col_f0, bmm2_col_f1, model_k//(bmm2_col_f0*bmm2_col_f1)],
        k: [bmm2_reduce_f0, bmm2_reduce_f1, seq_len//(bmm2_reduce_f0*bmm2_reduce_f1)]
    }
    root_tree = memory_level_tree_tiling(root_tree, tBmm2, tBmm2_tile)
    # tile tDiv
    # manually infer-bound
    b, h, m, n = tDiv.init.all_loops()
    b_bound = bmm2_batch_f0 * bmm2_batch_f1
    h_bound = bmm2_head_f0 * bmm2_head_f1
    m_bound = bmm2_row_f0 * bmm2_row_f1
    n_bound = seq_len
    # div factors
    b_f0 = 1
    h_f0 = 1
    m_f0 = 1
    n_f0 = seq_len
    tDiv_tile = {
        b: [b_f0, b_bound//b_f0],
        h: [h_f0, h_bound//h_f0],
        m: [m_f0, m_bound//m_f0],
        n: [n_f0, n_bound//n_f0]
    }
    root_tree = memory_level_tree_tiling(root_tree, tDiv, tDiv_tile)
    # tile tSum
    # manually infer-bound
    b, h, m, n = tSum.updates[0].all_loops()
    b_bound = bmm2_batch_f0 * bmm2_batch_f1
    h_bound = bmm2_head_f0 * bmm2_head_f1
    m_bound = bmm2_row_f0 * bmm2_row_f1
    n_bound = seq_len
    # div factors
    b_f0 = 1
    h_f0 = 1
    m_f0 = 1
    n_f0 = seq_len
    tSum_tile = {
        b: [b_f0, b_bound//b_f0],
        h: [h_f0, h_bound//h_f0],
        m: [m_f0, m_bound//m_f0],
        n: [n_f0, n_bound//n_f0]
    }
    root_tree = memory_level_tree_tiling(root_tree, tSum, tSum_tile)
    # tile tExp
    # manually infer-bound
    b, h, m, n = tExp.init.all_loops()
    b_bound = bmm2_batch_f0 * bmm2_batch_f1
    h_bound = bmm2_head_f0 * bmm2_head_f1
    m_bound = bmm2_row_f0 * bmm2_row_f1
    n_bound = seq_len
    # div factors
    b_f0 = 1
    h_f0 = 1
    m_f0 = 1
    n_f0 = seq_len
    tExp_tile = {
        b: [b_f0, b_bound//b_f0],
        h: [h_f0, h_bound//h_f0],
        m: [m_f0, m_bound//m_f0],
        n: [n_f0, n_bound//n_f0]
    }
    root_tree = memory_level_tree_tiling(root_tree, tExp, tExp_tile)
    # tile tSub
    # manually infer-bound
    b, h, m, n = tSub.init.all_loops()
    b_bound = bmm2_batch_f0 * bmm2_batch_f1
    h_bound = bmm2_head_f0 * bmm2_head_f1
    m_bound = bmm2_row_f0 * bmm2_row_f1
    n_bound = seq_len
    # div factors
    b_f0 = 1
    h_f0 = 1
    m_f0 = 1
    n_f0 = seq_len
    tSub_tile = {
        b: [b_f0, b_bound//b_f0],
        h: [h_f0, h_bound//h_f0],
        m: [m_f0, m_bound//m_f0],
        n: [n_f0, n_bound//n_f0]
    }
    root_tree = memory_level_tree_tiling(root_tree, tSub, tSub_tile)
    # tile tMax
    # manually infer-bound
    b, h, m, n = tMax.updates[0].all_loops()
    b_bound = bmm2_batch_f0 * bmm2_batch_f1
    h_bound = bmm2_head_f0 * bmm2_head_f1
    m_bound = bmm2_row_f0 * bmm2_row_f1
    n_bound = seq_len
    # div factors
    b_f0 = 1
    h_f0 = 1
    m_f0 = 1
    n_f0 = seq_len
    tMax_tile = {
        b: [b_f0, b_bound//b_f0],
        h: [h_f0, h_bound//h_f0],
        m: [m_f0, m_bound//m_f0],
        n: [n_f0, n_bound//n_f0]
    }
    root_tree = memory_level_tree_tiling(root_tree, tMax, tMax_tile)
    # tile tBmm1
    # manually infer-bound
    b, h, m, n, k = tBmm1.updates[0].all_loops()
    b_bound = bmm2_batch_f0 * bmm2_batch_f1
    h_bound = bmm2_head_f0 * bmm2_head_f1
    m_bound = bmm2_row_f0 * bmm2_row_f1
    n_bound = seq_len
    k_bound = model_k
    # div factors
    b_f0 = 1
    h_f0 = 1
    m_f0 = 1
    n_f0 = 16
    k_f0 = 16
    tBmm1_tile = {
        b: [b_f0, b_bound//b_f0],
        h: [h_f0, h_bound//h_f0],
        m: [m_f0, m_bound//m_f0],
        n: [n_f0, n_bound//n_f0],
        k: [k_f0, k_bound//k_f0]
    }
    root_tree = memory_level_tree_tiling(root_tree, tBmm1, tBmm1_tile)
    print_ir(root_tree.root)