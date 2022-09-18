#ifndef DOMINO_EXPR_H
#define DOMINO_EXPR_H

#include <fmt/ostream.h>
#include <general_visitor.h>
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
#undef X_DECL_EXPR
#undef X_DECL_EXPR_FINAL

class ExprNode {
 public:
  virtual ~ExprNode() = default;
};
// using Expr = Ref<ExprNode>;

struct BinOpNode : public ExprNode {
 public:
  std::string opt;
  Expr lhs, rhs;

  BinOpNode(std::string opt, Expr lhs, Expr rhs)
      : opt(std::move(opt)), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};
// using BinOp = Ref<BinOpNode>;

class UnOpNode : public ExprNode {
 public:
  std::string opt;
  Expr opr;

  UnOpNode(std::string opt, Expr opr) : opt(std::move(opt)), opr(std::move(opr)) {}
};
// using UnOp = Ref<UnOpNode>;

class CallNode : public ExprNode {
 public:
  std::string opt;
  std::vector<Expr> oprs;

  CallNode(std::string opt, const std::vector<Expr> oprs) : opt(std::move(opt)), oprs(oprs) {}
};
// using Call = Ref<CallNode>;

class IfExprNode : public ExprNode {
 public:
  Expr cond, then_case, else_case;

  IfExprNode(Expr cond, Expr then_case, Expr else_case)
      : cond(cond), then_case(then_case), else_case(else_case) {}
};
// using IfExpr = Ref<IfExprNode>;

class IntImmNode : public ExprNode {
 public:
  int64_t val;

  IntImmNode(int64_t val) : val(val) {}
};
// using IntImm = Ref<IntImmNode>;

class FloatImmNode : public ExprNode {
 public:
  double val;

  FloatImmNode(double val) : val(val) {}
};
// using FloatImm = Ref<FloatImmNode>;

class VarNode : public ExprNode {
 public:
  std::string id;

  VarNode(std::string id) : id(std::move(id)) {}
};
// using Var = Ref<VarNode>;

#define X_DECL_EXPR_FINAL(X) , X##Node
using FinalExprTypes = std::tuple<void
#include <x_macro/expr.x.h>
                                  >;
#undef X_DECL_EXPR
#undef X_DECL_EXPR_FINAL

template <typename F>
class ExprFunctor;
template <typename R, typename... Args>
class ExprFunctor<R(Args...)>
    : public GeneralVisitor<ExprFunctor<R(Args...)>, ExprNode, FinalExprTypes, R(Args...)> {
 public:
#define X_DECL_EXPR_FINAL(X) \
  virtual R ImplVisit(X, Args...) { throw std::runtime_error("not implemented"); }
#include <x_macro/expr.x.h>
#undef X_DECL_EXPR
#undef X_DECL_EXPR_FINAL
};

}  // namespace domino

#endif
