from dominoc import ir


__all__ = ["print_ir"]

def print_ir(tree, print_out=True):
    ret = ir.print_ir(tree)
    if print_out:
        print(ret)
    else:
        return ret
