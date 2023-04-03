from domino.program_ir import *


def self_attention(batch, num_heads, M, N, K, L):
    def bmm(Batch, B, M, N, K, tA, tB):
        batch = SLoop(Batch)
        b, m, n = [SLoop(x) for x in [B, M, N]]
        k = RLoop(K)
        tC = Tensor([Batch, B, M, N], dtype="int8")
        tC.Init([batch, b, m, n], const(0, "int8"))
        tC.Update([batch, b, m, n], tA[batch, b, m, k] *
                  tB[batch, b, n, k], op=ReduceOp("sum"), reduce_axis=[k])
        return tC

    def maxv(Batch, B, M, N, tA):
        batch = SLoop(Batch)
        b, m = [SLoop(x) for x in [B, M]]
        n = RLoop(N)
        tB = Tensor([Batch, B, M], dtype="int8")
        tB.Init([batch, b, m], const(-128, "int8"))
        tB.Update([batch, b, m], tA[batch, b, m, n],
                  op=ReduceOp("max"), reduce_axis=[n])
        return tB

    def sub(Batch, B, M, N, tA, tB):
        batch = SLoop(Batch)
        b, m, n = [SLoop(x) for x in [B, M, N]]
        tC = Tensor([Batch, B, M, N], dtype="int8")
        tC[batch, b, m, n] = tA[batch, b, m, n] - tB[batch, b, m]
        return tC

    def exp(Batch, B, M, N, tA):
        batch = SLoop(Batch)
        b, m, n = [SLoop(x) for x in [B, M, N]]
        tB = Tensor([Batch, B, M, N], dtype="int8")
        tB[batch, b, m, n] = Call(
            "int8", "exp", [tA[batch, b, m, n].as_expr()])
        return tB

    def sumv(Batch, B, M, N, tA):
        batch = SLoop(Batch)
        b, m = [SLoop(x) for x in [B, M]]
        n = RLoop(N)
        tB = Tensor([Batch, B, M], dtype="int8")
        tB.Init([batch, b, m], const(0, "int8"))
        tB.Update([batch, b, m], tA[batch, b, m, n],
                  op=ReduceOp("sum"), reduce_axis=[n])
        return tB

    def div(Batch, B, M, N, tA, tB):
        batch = SLoop(Batch)
        b, m, n = [SLoop(x) for x in [B, M, N]]
        tC = Tensor([Batch, B, M, N], dtype="int8")
        tC[batch, b, m, n] = tA[batch, b, m, n] / tB[batch, b, m]
        return tC

    def softmax(Batch, B, M, N, tA):
        T1 = maxv(Batch, B, M, N, tA)
        T2 = sub(Batch, B, M, N, tA, T1)
        T3 = exp(Batch, B, M, N, T2)
        T4 = sumv(Batch, B, M, N, T3)
        T5 = div(Batch, B, M, N, T3, T4)
        return T5

    A = Tensor([batch, num_heads, M, K], name="A", dtype="int8")
    B = Tensor([batch, num_heads, L, K], name="B", dtype="int8")
    C = Tensor([batch, num_heads, N, L], name="C", dtype="int8")

    T1 = bmm(batch, num_heads, M, L, K, A, B)
    T2 = softmax(batch, num_heads, M, L, T1)
    T3 = bmm(batch, num_heads, M, N, L, T2, C)
    return T3


def test_create_tree():
    batch = 64
    num_heads = 12
    hidden = 768
    seq_len = 1024
    model_k = hidden//num_heads
    T = self_attention(batch, num_heads, seq_len, model_k, model_k, seq_len)
    graph = make_prod_consum_graph(T)
    
    tensors = graph.nodes
    print("Tensors:")
    print([t.name for t in tensors])
    Q, K, Bmm1, Max, Sub, Exp, Sum, Div, V, Bmm2 = tensors
    
    memory_levels = [0, 1, 2]
    trees = [create_tree(t, memory_levels) for t in tensors]


if __name__ == "__main__":
    test_create_tree()
