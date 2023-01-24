from mculib import gen_batch_matmul_chain, gen_batch_matmul_chain_golden, gen_test_file


def test_gen_batch_matmul_chain():
    kernel, tensors, scalars = gen_batch_matmul_chain()
    golden, tensors, scalars = gen_batch_matmul_chain_golden()
    batch, M, N, K, L, input_offset, inter_offset1, inter_offset2, output_offset, clip_min, clip_max = [4, 32, 32, 32, 32, 0, 0, 0, 0, 0, 127]
    values = [batch, M, N, K, L, input_offset, inter_offset1, inter_offset2, output_offset, clip_min, clip_max]
    string = gen_test_file(kernel, golden, tensors, scalars, values)
    with open("tmp.cpp", "w") as fout:
        fout.write(string)


if __name__ == "__main__":
    test_gen_batch_matmul_chain()