#ifndef DOMINO_TRANSFORM_H
#define DOMINO_TRANSFORM_H

#include <expr.h>
#include <mutator.h>
#include <right_thread.h>

#include <unordered_map>

namespace domino {

Range InferBound(Expr expr);  // 总入口函数
SetGeneral TransformFirst(Expr expr);

SetGeneral SetConstAdd(SetGeneral a, SetGeneral b);
SetGeneral SetVarAdd(SetGeneral a, SetGeneral b);
SetGeneral SetConstMul(SetGeneral a, SetGeneral b);
SetGeneral SetVarMul(SetGeneral a, SetGeneral b);

Range HandleSetConst(SetConst sc);
Range HandleSetVar(SetVar sv);
Range range_div(Range divisor, Range dividend);

long long int compute_expr(Expr expr, std::unordered_map<std::string, long long int>* umap);
long long int compute_range(Range range, std::unordered_map<std::string, long long int>* umap,
                            int n);

class ExprTransformer : public IRFunctor<SetGeneral()> {
 public:
  SetGeneral ImplVisit(ConstInt op) override { return SetConst::make(op); }

  SetGeneral ImplVisit(Iterator op) override { return SetConst::make(op); }

  SetGeneral ImplVisit(Var op) override { return SetVar::make(op); }

  SetGeneral ImplVisit(Add op) override {
    auto lhs = Visit(op->a), rhs = Visit(op->b);
    if (lhs->stype == SET_CONST && rhs->stype == SET_CONST) return SetConstAdd(lhs, rhs);
    if (lhs->stype == SET_VAR) return SetVarAdd(lhs, rhs);
    return SetVarAdd(rhs, lhs);
  }

  SetGeneral ImplVisit(Sub op) override {
    auto lhs = Visit(op->a), rhs = Visit(op->b);
    if (rhs->stype == SET_CONST) {
      rhs.as<SetConstNode>()->negate();
      if (lhs->stype == SET_CONST) return SetConstAdd(lhs, rhs);
      return SetVarAdd(lhs, rhs);
    }
    rhs.as<SetVarNode>()->negate();
    return SetVarAdd(rhs, lhs);
  }

  SetGeneral ImplVisit(Neg op) override {
    auto hs = Visit(op->a);
    if (hs->stype == SET_CONST)
      hs.as<SetConstNode>()->negate();
    else
      hs.as<SetVarNode>()->negate();
    return hs;
  }

  SetGeneral ImplVisit(Mul op) override {
    auto lhs = Visit(op->a), rhs = Visit(op->b);
    if (lhs->stype == SET_CONST && rhs->stype == SET_CONST) return SetConstMul(lhs, rhs);
    if (lhs->stype == SET_VAR) return SetVarMul(lhs, rhs);
    return SetVarMul(rhs, lhs);
  }

  int iterator_name;
  SetGeneral ImplVisit(FloorDiv op) override {
    auto lhs = Visit(op->a), rhs = Visit(op->b);
    // 分子和分母禁止出现variable
    // 分子可以是任意范围，但分母不可以有0的可能性
    ASSERT(lhs->stype == SET_CONST && rhs->stype == SET_CONST);
    Range res =
        range_div(HandleSetConst(lhs.as<SetConstNode>()), HandleSetConst(rhs.as<SetConstNode>()));
    DType t = DType::make("ignore");
    Iterator it =
        Iterator::make(Var::make(t, " " + std::to_string(iterator_name)), res, IterTypeKind(0));
    iterator_name++;
    return SetConst::make(it);
  }
};

int compute_op(Expr expr) {
  if (expr.as<AddNode>().defined())
    return 1 + compute_op(expr.as<AddNode>()->a) + compute_op(expr.as<AddNode>()->b);
  if (expr.as<SubNode>().defined())
    return 1 + compute_op(expr.as<SubNode>()->a) + compute_op(expr.as<SubNode>()->b);
  if (expr.as<MulNode>().defined())
    return 1 + compute_op(expr.as<MulNode>()->a) + compute_op(expr.as<MulNode>()->b);
  if (expr.as<NegNode>().defined()) return 1 + compute_op(expr.as<NegNode>()->a);
  if (expr.as<FloorDivNode>().defined())
    return 1 + compute_op(expr.as<FloorDivNode>()->a) + compute_op(expr.as<FloorDivNode>()->b);
  return 0;
}

int compute_leaf(Expr expr) {
  if (expr.as<ConstIntNode>().defined()) return 1;
  if (expr.as<IteratorNode>().defined()) return 1;
  if (expr.as<VarNode>().defined()) return 1;
  if (expr.as<AddNode>().defined())
    return compute_leaf(expr.as<AddNode>()->a) + compute_leaf(expr.as<AddNode>()->b);
  if (expr.as<SubNode>().defined())
    return compute_leaf(expr.as<SubNode>()->a) + compute_leaf(expr.as<SubNode>()->b);
  if (expr.as<MulNode>().defined())
    return compute_leaf(expr.as<MulNode>()->a) + compute_leaf(expr.as<MulNode>()->b);
  if (expr.as<NegNode>().defined()) return compute_leaf(expr.as<NegNode>()->a);
  ASSERT(expr.as<FloorDivNode>().defined());
  return compute_leaf(expr.as<FloorDivNode>()->a) + compute_leaf(expr.as<FloorDivNode>()->b);
}

int compute_op_range(Range range, int n) {
  if (n == 0) return compute_op(range->beg);
  return compute_op(range->extent);
}

int compute_leaf_range(Range range, int n) {
  if (n == 0) return compute_leaf(range->beg);
  return compute_leaf(range->extent);
}

}  // namespace domino

#endif  // DOMINO_SIMPLIFY_H