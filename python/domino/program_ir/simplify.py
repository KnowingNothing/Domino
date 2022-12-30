from dominoc import ir
from .scalar_expr import _to_expr

__all__ = ["expr_simplify_match_pattern",
           "get_expr_simplify_match_patterns", "substitute_expr"]

expr_simplify_match_pattern = ir.expr_simplify_match_pattern
get_expr_simplify_match_patterns = ir.get_expr_simplify_match_patterns


def substitute_expr(expr, mapping):
    mapping = {k: _to_expr(v) for k, v in mapping.items()}
    expr = _to_expr(expr)
    return ir.substitute_expr(expr, mapping)
