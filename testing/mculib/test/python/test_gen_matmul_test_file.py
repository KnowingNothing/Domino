from mculib import gen_matmul, gen_matmul_golden, gen_test_file


def test_gen_matmul():
    kernel, tensors, scalars = gen_matmul()
    golden, tensors, scalars = gen_matmul_golden()
    M, N, K, input_offset, output_offset, clip_min, clip_max = [32, 32, 32, 0, 0, 0, 127]
    values = [M, N, K, input_offset, output_offset, clip_min, clip_max]
    string = gen_test_file(kernel, golden, tensors, scalars, values)
    with open("tmp.cpp", "w") as fout:
        fout.write(string)


if __name__ == "__main__":
    test_gen_matmul()