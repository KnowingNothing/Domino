#include <fmt/core.h>
#include <fmt/ranges.h>
#include <printer.h>

#include <vector>

namespace domino {

class IRPrinter : public MultiVisitors<ExprFunctor<std::string()>> {
 public:
  std::string ImplVisit(BinOp expr) override {
    return fmt::format("({} {} {})", Visit(expr->lhs), expr->opt, Visit(expr->rhs));
  }

  std::string ImplVisit(UnOp expr) override {
    return fmt::format("({}{})", expr->opt, Visit(expr->opr));
  }

  std::string ImplVisit(Call expr) override {
    std::vector<std::string> args;
    for (auto x : expr->oprs) {
      args.push_back(Visit(x));
    }
    return fmt::format("{}({})", expr->opt, fmt::join(args, ", "));
  }

  std::string ImplVisit(IfExpr expr) override {
    return fmt::format("({} ? {} : {})", Visit(expr->cond), Visit(expr->then_case),
                       Visit(expr->else_case));
  }

  std::string ImplVisit(IntImm expr) override { return fmt::format("{}", expr->val); }

  std::string ImplVisit(FloatImm expr) override { return fmt::format("{}", expr->val); }

  std::string ImplVisit(Var expr) override { return expr->id; }
};

std::ostream& operator<<(std::ostream& os, Expr expr) { return os << IRPrinter().Visit(expr); }

std::string to_string(Expr expr) { return IRPrinter().Visit(expr); }

}  // namespace domino