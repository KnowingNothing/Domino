#ifndef DOMINO_EXPR_H
#define DOMINO_EXPR_H

#include <ir_base.h>
#include <ref.h>

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace domino {

/// Don't use X_Macro for reference declaration
/// for better debug experience

// #define X_DECL_EXPR(X) \
//   class X##Node;       \
//   using X = Ref<X##Node>;
// #include <x_macro/expr.x.h>

class ExprNode : public IRBaseNode {
 public:
  virtual bool IsConst() const { return false; }
};

using Expr = Ref<ExprNode>;

struct BinOpNode : public ExprNode {
 public:
  std::string opt;
  Expr lhs, rhs;

  BinOpNode(std::string opt, Expr lhs, Expr rhs)
      : opt(std::move(opt)), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

using BinOp = Ref<BinOpNode>;

class UnOpNode : public ExprNode {
 public:
  std::string opt;
  Expr opr;

  UnOpNode(std::string opt, Expr opr) : opt(std::move(opt)), opr(std::move(opr)) {}
};

using UnOp = Ref<UnOpNode>;

class CallNode : public ExprNode {
 public:
  std::string func;
  std::vector<Expr> args;

  CallNode(std::string func, std::vector<Expr> args)
      : func(std::move(func)), args(std::move(args)) {}
};

using Call = Ref<CallNode>;

class IfExprNode : public ExprNode {
 public:
  Expr cond, then_case, else_case;

  IfExprNode(Expr cond, Expr then_case, Expr else_case)
      : cond(std::move(cond)), then_case(std::move(then_case)), else_case(std::move(else_case)) {}
};

using IfExpr = Ref<IfExprNode>;

class LoadNode : public ExprNode {
 public:
  std::string id;
  std::vector<Expr> indices;

  LoadNode(std::string id, std::vector<Expr> indices)
      : id(std::move(id)), indices(std::move(indices)) {}
};

using Load = Ref<LoadNode>;

class RangeNode : public ExprNode {
 public:
  Expr beg, extent, step;

  RangeNode(Expr beg, Expr extent, Expr step) : beg(beg), extent(extent), step(step) {}
};

using Range = Ref<RangeNode>;

class SliceNode : public ExprNode {
 public:
  std::string id;
  std::vector<Range> indices;

  SliceNode(std::string id, std::vector<Range> indices)
      : id(std::move(id)), indices(std::move(indices)) {}
};

using Slice = Ref<SliceNode>;

class ConstNode : public ExprNode {
 public:
  bool IsConst() const override { return true; }
};

using Const = Ref<ConstNode>;

class IntConstNode : public ConstNode {
 public:
  int64_t val;

  IntConstNode(int64_t val) : val(val) {}
};

using IntConst = Ref<IntConstNode>;

class FloatConstNode : public ConstNode {
 public:
  double val;

  FloatConstNode(double val) : val(val) {}
};

using FloatConst = Ref<FloatConstNode>;

class VarNode : public ExprNode {
 public:
  std::string id;

  VarNode(std::string id) : id(std::move(id)) {}
};

using Var = Ref<VarNode>;

class IteratorNode : public ExprNode {
 public:
  Var var;
  Range range;

  IteratorNode(Var var, Range range) : var(var), range(range) {}
};

using Iterator = Ref<IteratorNode>;

}  // namespace domino

#endif
