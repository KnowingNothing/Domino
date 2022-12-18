#include <fmt/core.h>
#include <fmt/ranges.h>
#include <ir_base.h>
#include <ir_functor.h>

#include <range/v3/view.hpp>
#include <vector>

namespace domino {

class IRStringifier : public IRFunctor<std::string()> {
 protected:
  std::string ImplVisit(BinOp expr) override {
    return fmt::format("({} {} {})", Visit(expr->lhs), expr->opt, Visit(expr->rhs));
  }

  std::string ImplVisit(UnOp expr) override {
    return fmt::format("({}{})", expr->opt, Visit(expr->opr));
  }

  std::string ImplVisit(Call expr) override {
    auto args = expr->args | ranges::view::transform([&](Expr e) { return Visit(e); });
    return fmt::format("{}({})", expr->func, fmt::join(args, ", "));
  }

  std::string ImplVisit(IfExpr expr) override {
    return fmt::format("({} ? {} : {})", Visit(expr->cond), Visit(expr->then_case),
                       Visit(expr->else_case));
  }

  std::string ImplVisit(IntConst expr) override { return fmt::format("{}", expr->val); }

  std::string ImplVisit(FloatConst expr) override { return fmt::format("{}", expr->val); }

  std::string ImplVisit(Var expr) override { return expr->id; }

  std::string ImplVisit(Load expr) override {
    auto indices = expr->indices | ranges::views::transform([&](Expr e) { return Visit(e); });
    return fmt::format("{}[{}]", expr->id, fmt::join(indices, ", "));
  }
};

std::ostream& operator<<(std::ostream& os, IRBase ir) { return os << IRStringifier().Visit(ir); }

std::string repr(IRBase ir) { return IRStringifier().Visit(ir); }

}  // namespace domino
