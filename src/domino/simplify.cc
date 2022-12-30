#include <simplify.h>

namespace domino {

bool ExprSimplifyMatchPattern(Expr target, Expr pattern) {
  ExprSimplifyPatternMatcher matcher;
  return matcher(target, pattern);
}

std::unordered_map<Var, Expr> GetExprSimplifyMatchPatterns(Expr target, Expr pattern) {
  ExprSimplifyPatternMatcher matcher;
  bool match = matcher(target, pattern);
  if (match) {
    return matcher.get_mapping();
  } else {
    return std::unordered_map<Var, Expr>();
  }
}

Expr SubstituteExpr(Expr expr, std::unordered_map<Var, Expr> mapping) {
  ExprSubstituter suber(mapping);
  return suber(expr);
}

}  // namespace domino