from domino.program_ir import *


def broadcast_to_s16x2(ctx, value: Var):
    value_s16 = ctx.map_var(
        f"{value.id.value}_s16", cast("int16", value))
    packed_value = ctx.map_var(f"pack_{value_s16.id.value}", pack_value(
        "int32", [value_s16, value_s16]))
    return packed_value


def alloc_register_array(ctx, shape, dtype, name, init_value):
    def helper(s, n):
        if (len(s) == 0):
            return ctx.map_var(f"{n}", make_const(init_value, dtype))
        else:
            return [helper(s[1:], n + f"{i}") for i in range(s[0])]
    return helper(shape, name)


def vload_to_register_array(ctx, name, array, length):
    return [ctx.map_var(f"{name}_{i}", array[i]) for i in range(length)]
