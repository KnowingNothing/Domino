#ifndef DOMINO_SIMPLIFY_H
#define DOMINO_SIMPLIFY_H

#include <ir_functor.h>
#include <logging/logging.h>

#include <unordered_map>
#include <vector>

namespace domino {
using namespace logging;

class ExprSimplifyPattern {
 public:
  ExprSimplifyPattern(Expr old, Expr replace) : old(std::move(old)), replace(std::move(replace)) {}
  Expr old;
  Expr replace;
};

/**
 * \brief Functor for expression pattern matching
 *
 * When matching two expressions, the left side is target expression, the right side is pattern
 * expression Rules:
 * 1. Any left Expr matches with right Var
 * 2. Except rule 1, different IR nodes don't match
 */
class ExprSimplifyPatternMatcher : public IRFunctor<bool(Expr)> {
 protected:
#define GENERAL_VISIT(OP)                                \
  if (!other.defined() || (op->dtype != other->dtype)) { \
    return false;                                        \
  }                                                      \
  Var as_var = other.as<VarNode>();                      \
  if (as_var.defined()) {                                \
    add_mapping(as_var, op);                             \
    return true;                                         \
  }                                                      \
  OP as_op = other.as<OP##Node>();                       \
  if (!as_op.defined()) {                                \
    return false;                                        \
  }

  bool ImplVisit(MemRef op, Expr other) override {
    GENERAL_VISIT(MemRef)
    return Visit(op->var, as_op->var) && Visit(op->offset, as_op->offset);
  }

#define VISIT_BIN(OP)                                        \
  bool ImplVisit(OP op, Expr other) override {               \
    GENERAL_VISIT(OP)                                        \
    return Visit(op->a, as_op->a) && Visit(op->b, as_op->b); \
  }

  VISIT_BIN(Add)
  VISIT_BIN(Sub)
  VISIT_BIN(Mul)
  VISIT_BIN(Div)
  VISIT_BIN(Mod)
  VISIT_BIN(FloorDiv)
  VISIT_BIN(FloorMod)
  VISIT_BIN(And)
  VISIT_BIN(Or)
  VISIT_BIN(XOr)
  VISIT_BIN(BitAnd)
  VISIT_BIN(BitOr)
  VISIT_BIN(BitXOr)
  VISIT_BIN(GT)
  VISIT_BIN(GE)
  VISIT_BIN(LT)
  VISIT_BIN(LE)
  VISIT_BIN(EQ)
  VISIT_BIN(NE)

#undef VISIT_BIN

  bool ImplVisit(Cast op, Expr other) override {
    GENERAL_VISIT(Cast)
    return Visit(op->a, as_op->a);
  }

  bool ImplVisit(Broadcast op, Expr other) override {
    GENERAL_VISIT(Broadcast)
    return Visit(op->a, as_op->a);
  }

  bool ImplVisit(Neg op, Expr other) override {
    GENERAL_VISIT(Neg)
    return Visit(op->a, as_op->a);
  }

  bool ImplVisit(Not op, Expr other) override {
    GENERAL_VISIT(Not)
    return Visit(op->a, as_op->a);
  }

  bool ImplVisit(BitNot op, Expr other) override {
    GENERAL_VISIT(BitNot)
    return Visit(op->a, as_op->a);
  }

  bool ImplVisit(Ceil op, Expr other) override {
    GENERAL_VISIT(Ceil)
    return Visit(op->a, as_op->a);
  }

  bool ImplVisit(Floor op, Expr other) override {
    GENERAL_VISIT(Floor)
    return Visit(op->a, as_op->a);
  }

  bool ImplVisit(Select op, Expr other) override {
    GENERAL_VISIT(Select)
    return Visit(op->a, as_op->a) && Visit(op->b, as_op->b) && Visit(op->c, as_op->c);
  }

  bool ImplVisit(Range op, Expr other) override {
    GENERAL_VISIT(Range)
    return Visit(op->beg, as_op->beg) && Visit(op->extent, as_op->extent) &&
           Visit(op->step, as_op->step);
  }

  bool ImplVisit(ExprList op, Expr other) override {
    GENERAL_VISIT(ExprList)
    int length = (int)op->value_list.size();
    if (length != (int)as_op->value_list.size()) {
      return false;
    }
    for (int i = 0; i < length; ++i) {
      if (!Visit(op->value_list[i], as_op->value_list[i])) {
        return false;
      }
    }
    return true;
  }

  bool ImplVisit(CondAll op, Expr other) override {
    GENERAL_VISIT(CondAll)
    return Visit(op->phases, as_op->phases);
  }

  bool ImplVisit(CondAny op, Expr other) override {
    GENERAL_VISIT(CondAny)
    return Visit(op->phases, as_op->phases);
  }

  bool ImplVisit(ConstInt op, Expr other) override {
    GENERAL_VISIT(ConstInt)
    return (op->value == as_op->value) && (op->dtype == as_op->dtype);
  }

  bool ImplVisit(ConstUInt op, Expr other) override {
    GENERAL_VISIT(ConstUInt)
    return (op->value == as_op->value) && (op->dtype == as_op->dtype);
  }

  bool ImplVisit(ConstFloat op, Expr other) override {
    GENERAL_VISIT(ConstFloat)
    return (op->value == as_op->value) && (op->dtype == as_op->dtype);
  }

  bool ImplVisit(ConstBFloat op, Expr other) override {
    GENERAL_VISIT(ConstBFloat)
    return (op->value == as_op->value) && (op->dtype == as_op->dtype);
  }

  bool ImplVisit(ConstTFloat op, Expr other) override {
    GENERAL_VISIT(ConstTFloat)
    return (op->value == as_op->value) && (op->dtype == as_op->dtype);
  }

  bool ImplVisit(ConstString op, Expr other) override {
    GENERAL_VISIT(ConstString)
    return op->value == as_op->value;
  }

  bool ImplVisit(Var op, Expr other) override {
    GENERAL_VISIT(Var)
    // can't reach here
    ASSERT(false) << "Reach undesirable point.";
    return false;
  }

  bool ImplVisit(Iterator op, Expr other) override {
    GENERAL_VISIT(Iterator)
    return Visit(op->var, as_op->var) && Visit(op->range, as_op->range) &&
           (op->iter_type == as_op->iter_type);
  }

  bool ImplVisit(NdLoad op, Expr other) override {
    GENERAL_VISIT(NdLoad)
    return Visit(op->mem_ref, as_op->mem_ref) && Visit(op->indices, as_op->indices);
  }

  bool ImplVisit(Load op, Expr other) override {
    GENERAL_VISIT(Load)
    return Visit(op->mem_ref, as_op->mem_ref) && Visit(op->addr, as_op->addr);
  }

  bool ImplVisit(MapVar op, Expr other) override {
    GENERAL_VISIT(MapVar)
    return Visit(op->var, as_op->var) && Visit(op->expr, as_op->expr);
  }

  bool ImplVisit(Slice op, Expr other) override {
    GENERAL_VISIT(Slice)
    int length = (int)op->indices.size();
    if (length != (int)as_op->indices.size()) {
      return false;
    }
    for (int i = 0; i < length; ++i) {
      if (!Visit(op->indices[i], as_op->indices[i])) {
        return false;
      }
    }
    return true;
  }

  bool ImplVisit(MemSlice op, Expr other) override {
    GENERAL_VISIT(MemSlice)
    return Visit(op->var, as_op->var) && Visit(op->offset, as_op->offset) &&
           Visit(op->slice, as_op->slice);
  }

  bool ImplVisit(Call op, Expr other) override {
    GENERAL_VISIT(Call)
    return Visit(op->func, as_op->func) && Visit(op->args, as_op->args);
  }

#undef GENERAL_VISIT
  void add_mapping(Var var, Expr expr) { mapping_results_[var] = expr; }

 private:
  std::unordered_map<Var, Expr> mapping_results_;

 public:
  std::unordered_map<Var, Expr> get_mapping() { return mapping_results_; }
};

/**
 * \brief Function that performs pattern matching
 * 
 * Left side is the target expression, right side is the matching pattern
 * Returns bool value indicating whether the matching succeeds.
 * The matching rules are from ExprSimplifyPatternMatcher
*/
bool ExprSimplifyMatchPattern(Expr target, Expr pattern);

class ExprSubstituter : public IRFunctor<Expr()> {
 protected:
  void addPattern(const ExprSimplifyPattern& pattern) { patterns_.push_back(pattern); }

 private:
  std::vector<ExprSimplifyPattern> patterns_;
};

class ExprSimplifier : public IRFunctor<Expr()> {
 protected:
  /// expressions
  Expr ImplVisit(MemRef mem_ref) override {
    Var new_var = mem_ref->var;
    Expr new_offset = Visit(mem_ref->offset);
    return MemRef::make(new_var, new_offset);
  }

  Expr ImplVisit(Add op) override {
    Expr new_a = Visit(op->a);
    Expr new_b = Visit(op->b);
    return Add::make(new_a, new_b);
  }

 private:
};
}  // namespace domino

#endif  // DOMINO_SIMPLIFY_H