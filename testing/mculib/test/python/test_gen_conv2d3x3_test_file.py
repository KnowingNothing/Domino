from mculib import gen_conv2d, gen_conv2d_golden, gen_test_file, gen_ring_buffer_test_file, gen_conv2d_ring_buffer


def test_gen_conv2d():
    kernel, tensors, scalars = gen_conv2d()
    golden, tensors, scalars = gen_conv2d_golden()
    N, H, W, inC, outC, input_offset, output_offset, clip_min, clip_max = [
        1, 32, 32, 16, 32, 0, 0, 0, 127]
    values = [N, H, W, inC, outC, input_offset,
              output_offset, clip_min, clip_max]
    string = gen_test_file(kernel, golden, tensors, scalars, values)
    with open("tmp.cpp", "w") as fout:
        fout.write(string)


def test_gen_conv2d_ring_buffer():
    kernel, tensors, scalars = gen_conv2d_ring_buffer()
    golden, tensors, scalars = gen_conv2d_golden()
    N, H, W, inC, outC, input_offset, output_offset, clip_min, clip_max = [
        1, 32, 32, 16, 32, 0, 0, 0, 127]
    values = [N, H, W, inC, outC, input_offset,
              output_offset, clip_min, clip_max]
    string = gen_ring_buffer_test_file(kernel, golden, tensors, [
                                       32*2, -1, 0, -1], scalars, values, 32*32*32+32*2)
    with open("tmp.cpp", "w") as fout:
        fout.write(string)


if __name__ == "__main__":
    test_gen_conv2d_ring_buffer()
