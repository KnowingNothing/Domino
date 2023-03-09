from dominoc import ir
from .scalar_expr import _to_expr
from .stmt import _to_stmt
from .block import _to_block

__all__ = ["expr_simplify_match_pattern",
           "get_expr_simplify_match_patterns",
           "substitute_expr", "simplify_expr", "substitute_ir", "simplify"]

expr_simplify_match_pattern = ir.expr_simplify_match_pattern
get_expr_simplify_match_patterns = ir.get_expr_simplify_match_patterns


def substitute_expr(expr, mapping):
    mapping = {k: _to_expr(v) for k, v in mapping.items()}
    expr = _to_expr(expr)
    return ir.substitute_expr(expr, mapping)


def substitute_stmt(ir_tree, mapping):
    mapping = {k: _to_expr(v) for k, v in mapping.items()}
    ir_tree = _to_stmt(ir_tree)
    return ir.substitute_stmt(ir_tree, mapping)


def substitute_block(ir_tree, mapping):
    mapping = {k: _to_expr(v) for k, v in mapping.items()}
    ir_tree = _to_block(ir_tree)
    return ir.substitute_block(ir_tree, mapping)


def substitute_ir(ir_tree, mapping):
    mapping = {k: _to_expr(v) for k, v in mapping.items()}
    return ir.substitute_ir(ir_tree, mapping)


def simplify_expr(expr):
    return ir.simplify_expr(expr)


def simplify(irm):
    return ir.simplify(irm)
