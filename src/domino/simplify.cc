#include <simplify.h>

namespace domino {
std::vector<ExprSimplifyPattern> ExprSimplifier::patterns_;

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
  auto suber = std::make_shared<ExprSubstituter>(mapping);
  StmtMutator mutator(suber);
  return mutator(stmt);
}

Block SubstituteBlock(Block block, std::unordered_map<Var, Expr> mapping) {
  auto suber = std::make_shared<ExprSubstituter>(mapping);
  auto stmt_mutator = std::make_shared<StmtMutator>(suber);
  BlockMutator block_mutator(stmt_mutator);
  return block_mutator(block);
}

Arch SubstituteArch(Arch arch, std::unordered_map<Var, Expr> mapping) {
  auto suber = std::make_shared<ExprSubstituter>(mapping);
  auto stmt_mutator = std::make_shared<StmtMutator>(suber);
  auto block_mutator = std::make_shared<BlockMutator>(stmt_mutator);
  ArchMutator arch_mutator(block_mutator);
  return arch_mutator(arch);
}

IRBase SubstituteIR(IRBase ir, std::unordered_map<Var, Expr> mapping) {
  if (ir.as<ExprNode>().defined()) {
    return SubstituteExpr(ir.as<ExprNode>(), mapping);
  } else if (ir.as<StmtNode>().defined()) {
    return SubstituteStmt(ir.as<StmtNode>(), mapping);
  } else if (ir.as<BlockNode>().defined()) {
    return SubstituteBlock(ir.as<BlockNode>(), mapping);
  } else if (ir.as<ArchNode>().defined()) {
    return SubstituteArch(ir.as<ArchNode>(), mapping);
  } else {
    throw std::runtime_error("The IR type is not supported in substitution.");
  }
}

Expr SimplifyExpr(Expr expr) {
  ExprSimplifier sim;
  return sim.Visit(expr);
}

IRBase Simplify(IRBase ir) {
  auto sim = std::make_shared<ExprSimplifier>();
  IRMutator mutator(sim);
  return mutator.Visit(ir);
}

}  // namespace domino