from domino.program_ir import *
from domino.analysis import *
import numpy as np


def test_te_loop():
    m = SLoop(32, "m")
    n = SLoop(32, "n")
    k = RLoop(32, "k")
    m0 = TLoop(2, "m")
    n0 = TLoop(2, "n")
    k0 = TLoop(16, "k")
    # print(m, n, k)
    # print(m0, n0, k0)


def test_te_tensor_access():
    m = SLoop(32, "m")
    n = SLoop(32, "n")
    A = Tensor([32, 32], name="A", dtype="int8")
    xx = A[m:m+4, n:n+4][1:2, ::2][0, 0]
    # print(xx)


def test_te_tensor_compute_1():
    m = SLoop(32, "m")
    n = SLoop(32, "n")
    A = Tensor([32, 32], name="A", dtype="int8")
    B = Tensor([32, 32], name="B", dtype="int8")
    C = Tensor([32, 32], name="C", dtype="int8")
    C[m, n] = const(0, "int8")
    C[m, n] = A[m, n] + B[m, n] * A[m, n]
    C.Update([m, n], A[m, n] * const(2, "int8"))
    tensors = C.updates[0].input_tensors()
    # print(tensors)
    assert tensors[0] == A
    assert tensors[1] == B


def test_te_tensor_compute_2():
    m = SLoop(32, "m")
    n = SLoop(32, "n")
    k = RLoop(32, "k")
    A = Tensor([32, 32], name="A", dtype="int8")
    B = Tensor([32, 32], name="B", dtype="int8")
    C = Tensor([32, 32], name="C", dtype="int8")
    C.Init([m, n], const(0, "int8"))
    C.Update([m, n], A[m, k] * B[k, n], op=ReduceOp("sum"), reduce_axis=[k])
    graph = make_prod_consum_graph(C)


def test_te_tensor_compute_3():
    m = SLoop(32, "m")
    n = SLoop(32, "n")
    k = RLoop(32, "k")
    x = SLoop(32, "x")
    y = SLoop(32, "y")
    z = RLoop(32, "z")
    A = Tensor([32, 32], name="A", dtype="int8")
    B = Tensor([32, 32], name="B", dtype="int8")
    C = Tensor([32, 32], name="C", dtype="int8")
    D = Tensor([32, 32], name="D", dtype="int8")
    E = Tensor([32, 32], name="E", dtype="int8")
    C.Init([m, n], const(0, "int8"))
    C.Update([m, n], A[m, k] * B[k, n], op=ReduceOp("sum"), reduce_axis=[k])
    E.Init([x, y], const(0, "int8"))
    E.Update([x, y], C[x, z] * D[z, y], op=ReduceOp("sum"), reduce_axis=[z])
    graph = make_prod_consum_graph(E)


def test_te_memory_level_tree_1():
    m = SLoop(32, "m")
    n = SLoop(32, "n")
    k = RLoop(32, "k")
    A = Tensor([32, 32], name="A", dtype="int8")
    B = Tensor([32, 32], name="B", dtype="int8")
    C = Tensor([32, 32], name="C", dtype="int8")
    C.Init([m, n], const(0, "int8"))
    C.Update([m, n], A[m, k] * B[k, n], op=ReduceOp("sum"), reduce_axis=[k])
    # C_tree = MemoryLevelTree()
    C_tree = MemoryLevelTree([0, 1, 2], C.var, {m.var: Range(0, 32, 1), n.var: Range(0, 32, 1), k.var: Range(0, 32, 1)})
    # C_tree.pretty_print()


def test_te_memory_level_tree_2():
    m = SLoop(32, "m")
    n = SLoop(32, "n")
    k = RLoop(32, "k")
    x = SLoop(32, "x")
    y = SLoop(32, "y")
    z = RLoop(32, "z")
    A = Tensor([32, 32], name="A", dtype="int8")
    B = Tensor([32, 32], name="B", dtype="int8")
    C = Tensor([32, 32], name="C", dtype="int8")
    D = Tensor([32, 32], name="D", dtype="int8")
    E = Tensor([32, 32], name="E", dtype="int8")
    C.Init([m, n], const(0, "int8"))
    C.Update([m, n], A[m, k] * B[k, n], op=ReduceOp("sum"), reduce_axis=[k])
    E.Init([x, y], const(0, "int8"))
    E.Update([x, y], C[x, z] * D[z, y], op=ReduceOp("sum"), reduce_axis=[z])
    levels = [0, 1, 2]
    C_tree = MemoryLevelTree(levels, C.var, {m.var: Range(0, 32, 1), n.var: Range(0, 32, 1), k.var: Range(0, 32, 1)})
    E_tree = MemoryLevelTree(levels, E.var, {x.var: Range(0, 32, 1), y.var: Range(0, 32, 1), z.var: Range(0, 32, 1)})

    merged_tree = E_tree.merge(C_tree, E.var, 2)
    # print_ir(merged_tree.root)


def self_attention():
    def bmm(B, M, N, K, tA, tB):
        b, m, n = [SLoop(x) for x in [B, M, N]]
        k = RLoop(K)
        tC = Tensor([B, M, N], dtype="int8")
        tC.Init([b, m, n], const(0, "int8"))
        tC.Update([b, m, n], tA[b, m, k] * tB[b, n, k], op=ReduceOp("sum"), reduce_axis=[k])
        return tC

    def maxv(B, M, N, tA):
        b, m = [SLoop(x) for x in [B, M]]
        n = RLoop(N)
        tB = Tensor([B, M], dtype="int8")
        tB.Init([b, m], const(-128, "int8"))
        tB.Update([b, m], tA[b, m, n], op=ReduceOp("max"), reduce_axis=[n])
        return tB

    def sub(B, M, N, tA, tB):
        b, m, n = [SLoop(x) for x in [B, M, N]]
        tC = Tensor([B, M, N], dtype="int8")
        tC[b, m, n] = tA[b, m, n] - tB[b, m]
        return tC

    def exp(B, M, N, tA):
        b, m, n = [SLoop(x) for x in [B, M, N]]
        tB = Tensor([B, M, N], dtype="int8")
        tB[b, m, n] = Call("int8", "exp", [tA[b, m, n].as_expr()])
        return tB

    def sumv(B, M, N, tA):
        b, m = [SLoop(x) for x in [B, M]]
        n = RLoop(N)
        tB = Tensor([B, M], dtype="int8")
        tB.Init([b, m], const(0, "int8"))
        tB.Update([b, m], tA[b, m, n], op=ReduceOp("sum"), reduce_axis=[n])
        return tB

    def div(B, M, N, tA, tB):
        b, m, n = [SLoop(x) for x in [B, M, N]]
        tC = Tensor([B, M, N], dtype="int8")
        tC[b, m, n] = tA[b, m, n] / tB[b, m]
        return tC

    def softmax(B, M, N, tA):
        T1 = maxv(B, M, N, tA)
        T2 = sub(B, M, N, tA, T1)
        T3 = exp(B, M, N, T2)
        T4 = sumv(B, M, N, T3)
        T5 = div(B, M, N, T3, T4)
        return T5

    batch, M, N, K, L = [2, 32, 16, 64, 32]
    A = Tensor([batch, M, K], name="A", dtype="int8")
    B = Tensor([batch, L, K], name="B", dtype="int8")
    C = Tensor([batch, N, L], name="C", dtype="int8")

    T1 = bmm(batch, M, L, K, A, B)
    T2 = softmax(batch, M, L, T1)
    T3 = bmm(batch, M, N, L, T2, C)
    return T3


def test_te_memory_level_tree_3():
    T3 = self_attention()
    ret = generate_merged_memory_level_trees(T3, [0, 1])

    tree = ret[9]

    graph = make_prod_consum_graph(T3)
    tensors = graph.nodes
    A, B, bmm1, maxv, sub, exp, sumv, div, C, bmm2 = tensors

    b, m, n, k = bmm2.updates[0].all_loops()

    tree = memory_level_tree_tiling(tree, bmm2, {
        b: [1, 2],
        m: [4, 8],
        n: [4, 4],
        k: [4, 8]
    })

    b, m, n = exp.init.all_loops()
    tree = memory_level_tree_tiling(tree, exp, {
        b: [4],
        m: [4],
        n: [4]
    })


def test_te_memory_level_tree_4():
    T3 = self_attention()
    trees = generate_merged_memory_level_trees(T3, [0, 1, 2])
    tree = trees[9]
    print_ir(tree.root)

    graph = make_prod_consum_graph(T3)
    tensors = graph.nodes
    A, B, bmm1, maxv, sub, exp, sumv, div, C, bmm2 = tensors

    num_levels = tree.get_available_levels(bmm2.var)
    tile_bmm2_space = generate_tile_tensor_computation_space(bmm2, bmm2.shape, num_levels)

    tile_bmm2_choice = {k: v[773%len(v)] for k, v in tile_bmm2_space.items()}
    print("Tile BMM2:")
    for k, v in tile_bmm2_choice.items():
        print(k, ":", v)
    
    tree = memory_level_tree_tiling(tree, bmm2, tile_bmm2_choice)
    print_ir(tree.root)
    for k, v in tree.var_map.items():
        print(k)
        print(v)
    for k, v in tree.bounds.items():
        print(k)
        print(v)
    infer_producer_shape(tree, bmm2)
    raise NotImplementedError()



if __name__ == "__main__":
    test_te_loop()
    test_te_tensor_access()
    test_te_tensor_compute_1()
    test_te_tensor_compute_2()
    test_te_tensor_compute_3()
    test_te_memory_level_tree_1()
    test_te_memory_level_tree_2()
    test_te_memory_level_tree_3()
    # test_te_memory_level_tree_4()
