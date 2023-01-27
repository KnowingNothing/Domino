from dominoc import passes

__all__ = ["flatten_array_access"]


def flatten_array_access(block, var_list, strides):
    return passes.flatten_array_access(block, var_list, strides)
