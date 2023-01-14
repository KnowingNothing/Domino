from domino.program_ir import *

def gen_mculib_header():
    return """
#include <matmul/matmul_mma_m2n2k16.h>
#include <matmul/matmul_mma_m2n4k16.h>

#include <cstdlib>

#include "mbed.h"

using namespace mculib;

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef float float32;
"""

def gen_test_file(kernel_to_test, kernel_golden, input_tensors, input_vars, concrete_vars):
    header = gen_mculib_header()
    left = "{"
    right = "}"
    endl = "\\n"

    assert len(input_vars) == len(concrete_vars)
    define_vars = [f"const {input_vars[i].dtype} {input_vars[i].id.value} = {concrete_vars[i]};\n" for i in range(len(input_vars))]
    define_vars_str = "".join(define_vars)

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
        tmp = []
        for s in tensor.shape:
            tmp.append(print_ir(s, print_out=False))
        lengths.append("*".join(tmp))
    define_arrays = [f"{t.dtype} {t.name}[{length}] = " + "{0};\n" for t, length in zip(input_tensors, lengths)]
    define_debug_arrays = [f"{t.dtype} {t.name}_debug[{length}] = " + "{0};\n" for t, length in zip(input_tensors, lengths)]
    init_arrays = [
        f"""
for (int _i = 0; _i < {length}; ++_i) {left}
    {t.name}[_i] = ({t.dtype})(rand() % 8);
#ifdef DEBUG
    {t.name}_debug[_i] = {t.name}[_i];
#endif
{right}\n
"""
    for t, length in zip(input_tensors, lengths)]

    check_arrays = [
        f"""
for (int _i = 0; _i < {length}; ++_i) {left}
    if ({t.name}_debug[_i] != {t.name}[_i])
        _errors += 1;
{right}\n
"""
    for t, length in zip(input_tensors, lengths)]
    
    
    
    return f"""
{header}

#define DEBUG 1

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
{kernel_golden.signature.kernel_name}({debug_call_args_str});

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