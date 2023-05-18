#include <logging/logging.h>
#include <right_thread.h>
#include <transform.h>

#include <algorithm>
#include <unordered_map>

namespace domino {

SetGeneral SetConstAdd(SetGeneral lhs, SetGeneral rhs) {
  ASSERT(lhs->stype == SET_CONST && rhs->stype == SET_CONST);
  SetConst a = lhs.as<SetConstNode>();
  SetConst b = rhs.as<SetConstNode>();
  a->Merge(b);
  return a;
}

SetGeneral SetVarAdd(SetGeneral lhs, SetGeneral rhs) {
  ASSERT(lhs->stype == SET_VAR);
  SetVar a = lhs.as<SetVarNode>();
  if (rhs->stype == SET_CONST) {
    a->cons_set->Merge(rhs.as<SetConstNode>());
    return a;  // 肯定不可能变成SetConst
  }
  a->Merge(rhs.as<SetVarNode>());
  if (a->ifConst()) return a->cons_set;
  return a;
}

SetGeneral SetConstMul(SetGeneral lhs, SetGeneral rhs) {
  ASSERT(lhs->stype == SET_CONST && rhs->stype == SET_CONST);
  SetConst a = lhs.as<SetConstNode>();
  SetConst b = rhs.as<SetConstNode>();
  a->Mul(b);
  return a;
}

SetGeneral SetVarMul(SetGeneral lhs, SetGeneral rhs) {
  ASSERT(lhs->stype == SET_VAR);
  SetVar a = lhs.as<SetVarNode>();
  if (rhs->stype == SET_CONST)
    a->Mul(rhs.as<SetConstNode>());
  else
    a->Mul(rhs.as<SetVarNode>());
  if (a->ifConst()) return a->cons_set;
  return a;
}

long long int ConstIntToValue(Expr e) {
  ASSERT(e.as<ConstIntNode>().defined());
  return (e.as<ConstIntNode>())->value;
}

Range range_div(Range divisor, Range dividend) {
  auto dividend_lower = ConstIntToValue(dividend->beg);
  auto dividend_upper = ConstIntToValue(dividend->extent) + dividend_lower - 1;
  ASSERT(dividend_upper >= dividend_lower);
  if (dividend_lower <= 0 && dividend_upper >= 0)
    throw std::runtime_error("The range of dividend includes 0!");
  auto divisor_lower = ConstIntToValue(divisor->beg);
  auto divisor_upper = ConstIntToValue(divisor->extent) + divisor_lower - 1;
  auto candidate = {divisor_lower / dividend_lower, divisor_lower / dividend_upper,
                    divisor_upper / dividend_lower, divisor_upper / dividend_upper};
  long long int beg = std::min(candidate), end = std::max(candidate);
  return Range::make(const_int(beg), const_int(end - beg + 1), const_int(1));
}

// Range: beg, extent, step
// TermSet<CoefNum, Iterator>
// T1 coef
// std::vector<T2> element
Range HandleSetConst(SetConst sc) {
  if (sc->ifNumber())
    return Range::make(const_int(sc->cons_int->value), const_int(1), const_int(1));

  long long int beg = sc->cons_int->value;
  long long int end = beg;

  std::vector<Term<CoefNum, Iterator>>& p = (sc->ts).terms;
  for (int i = 0; i < p.size(); ++i) {
    // 把p[i].element各项全乘起来
    ASSERT(!(p[i].coef->ifZero()));
    long long int term_beg = 1, term_end = 1;
    for (int j = 0; j < p[i].element.size(); ++j) {
      Iterator it = p[i].element[j];
      auto tmp_beg = ConstIntToValue(it->range->beg);
      auto tmp_end = ConstIntToValue(it->range->extent) + tmp_beg - 1;
      auto candidate = {term_beg * tmp_beg, term_beg * tmp_end, term_end * tmp_beg,
                        term_end * tmp_end};
      term_beg = std::min(candidate);
      term_end = std::max(candidate);
    }

    // 判断p[i].coef正负并乘上叠加到前一项上
    long long int coef = p[i].coef->value;
    if (coef > 0) {
      beg += coef * term_beg;
      end += coef * term_end;
    } else {
      beg += coef * term_end;
      end += coef * term_beg;
    }
  }
  ASSERT(end >= beg);
  return Range::make(const_int(beg), const_int(end - beg + 1), const_int(1));
}

// 假设Var全部都非负
Range HandleSetVar(SetVar sv) {
  if (sv->ifConst()) return HandleSetConst(sv->cons_set);

  Range res = HandleSetConst(sv->cons_set);

  std::vector<Term<SetConst, Var>>& p = (sv->ts).terms;
  for (int i = 0; i < p.size(); ++i) {
    // 把p[i].element各项全乘起来
    ASSERT(!(p[i].coef->ifZero()));
    // TODO: Expr和Var？？？
    Expr term = p[i].element[0];  // 是个Var
    for (int j = 1; j < p[i].element.size(); ++j) {
      Expr v = p[i].element[j];
      term = term * v;
    }

    // 乘上p[i].coef叠加到前一项上
    // 防止×0和+0和×±1的情况出现
    Range tmp = HandleSetConst(p[i].coef);
    long long int beg_value = ConstIntToValue(tmp->beg),
                  extent_value = ConstIntToValue(tmp->extent);
    if (beg_value != 0) {  // ×0
      Expr aug = term;
      if (beg_value == -1)  // ×-1
        aug = -aug;
      else if (beg_value != 1) {  // ×1
        Expr beg = const_int(beg_value);
        aug = aug * beg;
      }
      ConstInt check_0 = (res->beg).as<ConstIntNode>();
      if (check_0.defined() && check_0->value == 0)  // +0
        res->beg = aug;
      else
        res->beg = res->beg + aug;
    }
    if (extent_value != 1) {
      Expr aug = term;
      if (extent_value == 0)
        aug = -aug;
      else if (extent_value != 2) {
        Expr extent = const_int(extent_value - 1);
        aug = aug * extent;
      }
      ConstInt check_0 = (res->extent).as<ConstIntNode>();
      if (check_0.defined() && check_0->value == 0)  // +0
        res->extent = aug;
      else
        res->extent = res->extent + aug;
    }
  }
  return res;
}

SetGeneral TransformFirst(Expr expr) {
  ExprTransformer trans;
  trans.iterator_name = 0;
  return trans.Visit(expr);
}

Range InferBound(Expr expr) {
  SetGeneral simplified = TransformFirst(expr);

  // 最后代入所有的iteration
  if (simplified->stype == SET_CONST)
    return HandleSetConst(simplified.as<SetConstNode>());
  else {
    ASSERT(simplified->stype == SET_VAR);
    return HandleSetVar(simplified.as<SetVarNode>());
  }
}

long long int compute_expr(Expr expr, std::unordered_map<std::string, long long int>* umap) {
  if (expr.as<ConstIntNode>().defined()) return expr.as<ConstIntNode>()->value;
  if (expr.as<IteratorNode>().defined()) return (*umap)[expr.as<IteratorNode>()->getID()];
  if (expr.as<VarNode>().defined()) return (*umap)[expr.as<VarNode>()->getID()];
  if (expr.as<AddNode>().defined())
    return compute_expr(expr.as<AddNode>()->a, umap) + compute_expr(expr.as<AddNode>()->b, umap);
  if (expr.as<SubNode>().defined())
    return compute_expr(expr.as<SubNode>()->a, umap) - compute_expr(expr.as<SubNode>()->b, umap);
  if (expr.as<MulNode>().defined())
    return compute_expr(expr.as<MulNode>()->a, umap) * compute_expr(expr.as<MulNode>()->b, umap);
  if (expr.as<NegNode>().defined()) return -compute_expr(expr.as<NegNode>()->a, umap);
  ASSERT(expr.as<FloorDivNode>().defined());
  return compute_expr(expr.as<FloorDivNode>()->a, umap) /
         compute_expr(expr.as<FloorDivNode>()->b, umap);
}

long long int compute_range(Range range, std::unordered_map<std::string, long long int>* umap,
                            int n) {
  if (n == 0) return compute_expr(range->beg, umap);
  return compute_expr(range->extent, umap);
}
}  // namespace domino