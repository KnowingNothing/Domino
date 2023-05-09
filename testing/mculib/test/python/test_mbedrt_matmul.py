from domino.runtime.mbed import MbedRuntime
from mculib import gen_matmul, gen_matmul_golden
import numpy as np


def test_run_matmul():
    golden, tensors, scalars = gen_matmul_golden()
    kernel, tensors, scalars = gen_matmul()
    tA, tB, tC, tScales = tensors
    M, N, K, input_offset, output_offset, clip_min, clip_max = [
        32,
        32,
        32,
        0,
        0,
        0,
        127,
    ]
    values = [M, N, K, input_offset, output_offset, clip_min, clip_max]
    # rt = MbedRuntime("/dev/ttyACM0", "NUCLEO_H7A3ZI_Q")
    rt = MbedRuntime.from_target_name()

    rt.alloc_buffer_from_numpy(tA.name, np.ones([32, 32], dtype="int8"))
    rt.alloc_buffer_from_numpy(tB.name, np.ones([32, 32], dtype="int8"), const=True)
    rt.alloc_buffer_from_numpy(tC.name, np.ones([32, 32], dtype="int8"))
    rt.alloc_buffer_from_numpy(tScales.name, np.ones([32], dtype="float32"), const=True)

    rt.set_kernel("golden", golden)
    rt.set_kernel("kernel", kernel)

    args = [str(ts.name) for ts in tensors] + values
    print(args)
    rt.set_invoke("golden", [("golden", args)])
    rt.set_invoke("kernel", [("kernel", args)])
    rt.set_invoke("golden+kernel", [("golden", args), ("kernel", args)])

    rt.execute("golden")
    print(rt.get_buffer_numpy(tA.name))
    print(rt.get_buffer_numpy(tB.name))
    print(rt.get_buffer_numpy(tC.name))

    rt.execute("kernel")
    print(rt.get_buffer_numpy(tA.name))
    print(rt.get_buffer_numpy(tB.name))
    print(rt.get_buffer_numpy(tC.name))

    rt.execute("golden+kernel")
    print(rt.get_buffer_numpy(tA.name))
    print(rt.get_buffer_numpy(tB.name))
    print(rt.get_buffer_numpy(tC.name))


if __name__ == "__main__":
    test_run_matmul()
