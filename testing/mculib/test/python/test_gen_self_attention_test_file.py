from mculib import gen_self_attention, gen_self_attention_golden, gen_test_file


def test_gen_self_attention():
    kernel, tensors, scalars = gen_self_attention()
    golden, tensors, scalars = gen_self_attention_golden()
    [seq_len, hidden, num_heads, input_offset, matmul0_offset, matmul1_offset,
     matmul2_offset, bmm0_input_offset, bmm0_output_offset, softmax_input_offset,
     softmax_exp_offset, softmax_output_offset, bmm1_input_offset, bmm1_output_offset,
     clip_min, clip_max] = [
        32, 64, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -128, 127
    ]
    values = [seq_len, hidden, num_heads, input_offset, matmul0_offset, matmul1_offset,
              matmul2_offset, bmm0_input_offset, bmm0_output_offset, softmax_input_offset,
              softmax_exp_offset, softmax_output_offset, bmm1_input_offset, bmm1_output_offset,
              clip_min, clip_max]
    string = gen_test_file(kernel, golden, tensors, scalars, values, need_debug=True)
    with open("tmp.cpp", "w") as fout:
        fout.write(string)


if __name__ == "__main__":
    test_gen_self_attention()
