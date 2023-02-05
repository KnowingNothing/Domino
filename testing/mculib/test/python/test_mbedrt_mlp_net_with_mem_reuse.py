from domino.runtime.mbed import MbedRuntime
from mculib import gen_matmul, gen_matmul_golden
import numpy as np


def mlp_net_shapes(hidden_sizes, batch_size=1):
    wgt_shps = list(zip(hidden_sizes, hidden_sizes[1:]))
    act_shps = [(batch_size, h) for h in hidden_sizes]
    return wgt_shps, act_shps


def test_run_mlp_net():
    kernel, tensors, scalars = gen_matmul()
    # rt = MbedRuntime("/dev/ttyACM0", "NUCLEO_H7A3ZI_Q")
    rt = MbedRuntime.from_target_name()

    batch_size = 2
    hidden_sizes = [256, 128, 64, 32, 64, 128, 256]
    wgt_shps = list(zip(hidden_sizes[1:], hidden_sizes))
    act_shps = [(batch_size, h) for h in hidden_sizes]

    rt.set_kernel("matmul", kernel)

    for i, shp in enumerate(wgt_shps):
        np_arr = np.zeros(shp, dtype="int8")
        np_arr[:, :2] = 1
        rt.alloc_buffer_from_numpy(f"W{i}", np_arr, const=True)
        rt.alloc_buffer_from_numpy(
            f"scales{i}", np.ones([shp[0]], dtype="float32"), const=True
        )

    max_act_mem = max(
        int(np.prod(i)) + int(np.prod(o)) for i, o in zip(act_shps, act_shps[1:])
    )
    ping = True
    for i, shp in enumerate(act_shps):
        nbytes = int(np.prod(shp))
        rt.alloc_buffer(
            f"X{i}",
            nbytes,
            dtype="int8",
            offset=0 if ping else max_act_mem - nbytes,
        )
        ping = not ping

    calls = [
        (
            "matmul",
            [
                f"X{i}",
                f"W{i}",
                f"X{i+1}",
                f"scales{i}",
                batch_size,
                ho,
                hi,
                0,
                0,
                0,
                127,
            ],
        )
        for i, (ho, hi) in enumerate(wgt_shps)
    ]
    rt.set_invoke("mlp-net", calls)

    def infer(X):
        rt.set_buffer_data("X0", X)
        rt.execute("mlp-net")
        return rt.get_buffer_numpy(f"X{len(hidden_sizes) - 1}")

    X = np.empty(act_shps[0], dtype="int8")

    X[...] = 1
    Y = infer(X)
    assert np.all(Y == min(int(2 ** (len(hidden_sizes) - 1)), 127))
    X[...] = 2
    Y = infer(X)
    assert np.all(Y == min(int(2 ** (len(hidden_sizes))), 127))


if __name__ == "__main__":
    test_run_mlp_net()
