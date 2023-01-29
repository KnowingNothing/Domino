from domino.program_ir import *
from domino.type_system.dtype import DType
import random
from functools import reduce


def gen_mculib_header():
    return """
#include <matmul/matmul_mma_m2n2k16.h>
#include <matmul/matmul_mma_m2n4k16.h>

#include <cstdlib>
#include <cmath>

#include "mbed.h"

using namespace mculib;

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef float float32;
"""


def generate_random_value(length, dtype):
    dtype = DType.make(dtype)
    if dtype.is_int() or dtype.is_uint():
        min_v = dtype.min_limit()
        max_v = dtype.max_limit()
        return [random.randint(min_v, max_v) for i in range(length)]
    elif dtype.is_float():
        return [random.random() for i in range(length)]
    else:
        raise RuntimeError(
            f"Currently don't support random generation for {dtype}")


def gen_test_file(kernel_to_test, kernel_golden, input_tensors, input_vars, concrete_vars, need_debug=True):
    header = gen_mculib_header()
    left = "{"
    right = "}"
    endl = "\\n"

    assert len(input_vars) == len(
        concrete_vars), f"{len(input_vars)} vs {len(concrete_vars)}"
    define_vars = [
        f"const {input_vars[i].dtype} {input_vars[i].id.value} = {concrete_vars[i]};\n" for i in range(len(input_vars))]
    define_vars_str = "".join(define_vars)

    sub_map = {v: value for v, value in zip(input_vars, concrete_vars)}

    call_args = [
        f"&{t.name}[0]" for t in input_tensors
    ]

    debug_call_args = [
        f"&{t.name}_debug[0]" for t in input_tensors
    ]

    call_args.extend([f"{v.id.value}" for v in input_vars])
    call_args_str = ", ".join(call_args)

    debug_call_args.extend([f"{v.id.value}" for v in input_vars])
    debug_call_args_str = ", ".join(debug_call_args)

    lengths = []
    for tensor in input_tensors:
        # tmp = []
        length = 1
        for s in tensor.shape:
            new_s = simplify_expr(substitute_expr(s, sub_map))
            assert new_s.is_const()
            length *= new_s.value
            # tmp.append(print_ir(substitute_expr(s, sub_map), print_out=False))
        # lengths.append("*".join(tmp))
        # lengths.append(reduce(lambda a, b: a * b, tmp, 1))
        lengths.append(length)

    define_arrays = []
    define_debug_arrays = []
    for t, length in zip(input_tensors, lengths):
        if t.is_const():
            init_values = generate_random_value(length, t.dtype)
            init_values = ",".join(map(str, init_values))
            define_arrays.append(
                f"const {t.dtype} {t.name}[{length}] = " + "{" + f"{init_values}" + "};\n")
            define_debug_arrays.append(
                f"const {t.dtype} {t.name}_debug[{length}] = " + "{" + f"{init_values}" + "};\n")
        else:
            define_arrays.append(f"{t.dtype} {t.name}[{length}] = " + "{0};\n")
            define_debug_arrays.append(
                f"{t.dtype} {t.name}_debug[{length}] = " + "{0};\n")

    init_arrays = [
        f"""
for (int _i = 0; _i < {length}; ++_i) {left}
    {t.name}[_i] = ({t.dtype})(rand() % 8);
#ifdef DEBUG
    {t.name}_debug[_i] = {t.name}[_i];
#endif
{right}\n
"""
        if not t.is_const() else "" for t, length in zip(input_tensors, lengths)]

    check_arrays = [
        f"""
for (int _i = 0; _i < {length}; ++_i) {left}
    if ({t.name}_debug[_i] != {t.name}[_i])
        _errors += 1;
{right}\n
"""
        if not t.is_const() else "" for t, length in zip(input_tensors, lengths)]

    return f"""
{header}

{'' if need_debug else '//'} #define DEBUG 1

Timer _t;

{define_vars_str}

{kernel_to_test.gen_function()}

{"".join(define_arrays)}

#ifdef DEBUG
{kernel_golden.gen_function()}

{"".join(define_debug_arrays)}
#endif

int main() {left}

{"".join(init_arrays)}

_t.start();
{kernel_to_test.signature.kernel_name}({call_args_str});
_t.stop();

printf("The time taken was %llu milliseconds{endl}",
         std::chrono::duration_cast<std::chrono::milliseconds>(_t.elapsed_time()).count());

#ifdef DEBUG
_t.start();
{kernel_golden.signature.kernel_name}({debug_call_args_str});
_t.stop();

printf("The golden time taken was %llu milliseconds{endl}",
         std::chrono::duration_cast<std::chrono::milliseconds>(_t.elapsed_time()).count());

int _errors = 0;
{"".join(check_arrays)}

if (_errors == 0) {left}
    printf("Correctness check passed!{endl}");
{right} else {left}
    printf("Errors (%d)!{endl}", _errors);
{right}
#endif

return 0;
{right}

"""
