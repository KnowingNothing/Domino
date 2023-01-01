from dominoc import ir
from .simplify import simplify_expr


__all__ = ["print_ir"]

def print_ir(tree, print_out=True, simplify=True):
    if simplify:
        tree = simplify_expr(tree)
    ret = ir.print_ir(tree)
    if print_out:
        print(ret)
    else:
        return ret
