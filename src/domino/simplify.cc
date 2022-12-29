#include <simplify.h>

namespace domino {

bool ExprSimplifyMatchPattern(Expr target, Expr pattern) {
  ExprSimplifyPatternMatcher matcher;
  return matcher(target, pattern);
}

}  // namespace domino