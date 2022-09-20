#ifndef DOMINO_EXPR_H
#define DOMINO_EXPR_H

#include <ir_base.h>
#include <ref.h>

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace domino {

#define X_DECL_EXPR(X) \
  class X##Node;       \
  using X = Ref<X##Node>;
#include <x_macro/expr.x.h>

class ExprNode : public IRBaseNode {
 public:
  virtual bool IsConst() const { return false; }
};

struct BinOpNode : public ExprNode {
 public:
  std::string opt;
  Expr lhs, rhs;

  BinOpNode(std::string opt, Expr lhs, Expr rhs)
      : opt(std::move(opt)), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

class UnOpNode : public ExprNode {
 public:
  std::string opt;
  Expr opr;

  UnOpNode(std::string opt, Expr opr) : opt(std::move(opt)), opr(std::move(opr)) {}
};

class CallNode : public ExprNode {
 public:
  std::string func;
  std::vector<Expr> args;

  CallNode(std::string func, std::vector<Expr> args)
      : func(std::move(func)), args(std::move(args)) {}
};

class IfExprNode : public ExprNode {
 public:
  Expr cond, then_case, else_case;

  IfExprNode(Expr cond, Expr then_case, Expr else_case)
      : cond(std::move(cond)), then_case(std::move(then_case)), else_case(std::move(else_case)) {}
};

class LoadNode : public ExprNode {
 public:
  std::string id;
  std::vector<Expr> indices;

  LoadNode(std::string id, std::vector<Expr> indices)
      : id(std::move(id)), indices(std::move(indices)) {}
};

class RangeNode : public ExprNode {
 public:
  Expr beg, extent, step;

  RangeNode(Expr beg, Expr extent, Expr step) : beg(beg), extent(extent), step(step) {}
};

class SliceNode : public ExprNode {
 public:
  std::string id;
  std::vector<Range> indices;

  SliceNode(std::string id, std::vector<Range> indices)
      : id(std::move(id)), indices(std::move(indices)) {}
};

class ConstNode : public ExprNode {
 public:
  bool IsConst() const override { return true; }
};

class IntConstNode : public ConstNode {
 public:
  int64_t val;

  IntConstNode(int64_t val) : val(val) {}
};

class FloatConstNode : public ConstNode {
 public:
  double val;

  FloatConstNode(double val) : val(val) {}
};

class VarNode : public ExprNode {
 public:
  std::string id;

  VarNode(std::string id) : id(std::move(id)) {}
};

class IteratorNode : public ExprNode {
 public:
  Var var;
  Range range;

  IteratorNode(Var var, Range range) : var(var), range(range) {}
};

}  // namespace domino

#endif
