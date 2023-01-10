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
    return matcher.getMapping();
  } else {
    return std::unordered_map<Var, Expr>();
  }
}

Expr SubstituteExpr(Expr expr, std::unordered_map<Var, Expr> mapping) {
  ExprSubstituter suber(mapping);
  return suber(expr);
}

Stmt SubstituteStmt(Stmt stmt, std::unordered_map<Var, Expr> mapping) {
  ExprSubstituter suber(mapping);
  StmtMutator mutator(&suber);
  return mutator(stmt);
}

Block SubstituteBlock(Block block, std::unordered_map<Var, Expr> mapping) {
  ExprSubstituter suber(mapping);
  StmtMutator stmt_mutator(&suber);
  BlockMutator block_mutator(&stmt_mutator);
  return block_mutator(block);
}

IRBase SubstituteIR(IRBase ir, std::unordered_map<Var, Expr> mapping) {
  if (ir.as<ExprNode>().defined()) {
    return SubstituteExpr(ir.as<ExprNode>(), mapping);
  } else if (ir.as<StmtNode>().defined()) {
    return SubstituteStmt(ir.as<StmtNode>(), mapping);
  } else if (ir.as<BlockNode>().defined()) {
    return SubstituteBlock(ir.as<BlockNode>(), mapping);
  } else {
    throw std::runtime_error("The IR type is not supported in substitution.");
  }
}

Expr SimplifyExpr(Expr expr) {
  ExprSimplifier sim;
  return sim.Visit(expr);
}

}  // namespace domino