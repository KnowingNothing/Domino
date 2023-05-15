#ifndef DOMINO_TRANSFORM_H
#define DOMINO_TRANSFORM_H

#include <expr.h>
#include <mutator.h>
#include <right_thread.h>

#include <unordered_map>

namespace domino {

Range InferBound(Expr expr);  // 总入口函数
SetGeneral SetConstAdd(SetGeneral a, SetGeneral b);
SetGeneral SetVarAdd(SetGeneral a, SetGeneral b);
SetGeneral SetConstMul(SetGeneral a, SetGeneral b);
SetGeneral SetVarMul(SetGeneral a, SetGeneral b);
Range HandleSetConst(SetConst sc);
Range HandleSetVar(SetVar sv);
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
};

}  // namespace domino

#endif  // DOMINO_SIMPLIFY_H