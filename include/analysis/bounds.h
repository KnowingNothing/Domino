#ifndef DOMINO_ANALYSIS_BOUNDS_H
#define DOMINO_ANALYSIS_BOUNDS_H

#include <simplify.h>
#include <visitor.h>

namespace domino {

namespace analysis {

class InferExprBound : public IRVisitor<> {
 protected:
  void ImplVisitBinExpr(BinExpr op) {
    Visit(op->a);
    Visit(op->b);
    ASSERT(this->bounds_.count(op->a));
    ASSERT(this->bounds_.count(op->b));
    Range ra = this->bounds_[op->a];
    Range rb = this->bounds_[op->b];
    ConstInt as_int = ra->step.as<ConstIntNode>();
    ASSERT(as_int.defined() && as_int->value == 1);
    as_int = rb->step.as<ConstIntNode>();
    ASSERT(as_int.defined() && as_int->value == 1);
  }

  void ImplVisit(Add op) override {
    ImplVisitBinExpr(op);
    Range ra = this->bounds_[op->a];
    Range rb = this->bounds_[op->b];
    Expr beg = SimplifyExpr(ra->beg + rb->beg);
    Expr extent = SimplifyExpr(ra->extent + rb->extent - const_int(1));
    Expr step = const_int(1);
    Range bound = Range::make(beg, extent, step);
    this->bounds_[op] = bound;
  }

  void ImplVisit(Sub op) override {
    ImplVisitBinExpr(op);
    Range ra = this->bounds_[op->a];
    Range rb = this->bounds_[op->b];
    Expr beg = SimplifyExpr(ra->beg - (rb->beg + rb->extent - const_int(1)));
    Expr extent = SimplifyExpr(ra->extent + rb->extent - const_int(1));
    Expr step = const_int(1);
    Range bound = Range::make(beg, extent, step);
    this->bounds_[op] = bound;
  }

  void ImplVisit(Mul op) override {
    ImplVisitBinExpr(op);
    Range ra = this->bounds_[op->a];
    Range rb = this->bounds_[op->b];
    ConstInt as_int = op->a.as<ConstIntNode>();
    if (as_int.defined()) {
      if (as_int->value >= 0) {
        Expr beg = SimplifyExpr(op->a * rb->beg);
        Expr extent = SimplifyExpr(op->a * (rb->extent - const_int(1)) + const_int(1));
        Expr step = const_int(1);
        Range bound = Range::make(beg, extent, step);
        this->bounds_[op] = bound;
      } else {
        Expr beg = SimplifyExpr(op->a * (rb->beg + rb->extent - const_int(1)));
        Expr extent = SimplifyExpr((-op->a) * (rb->extent - const_int(1)) + const_int(1));
        Expr step = const_int(1);
        Range bound = Range::make(beg, extent, step);
        this->bounds_[op] = bound;
      }
    } else {
      as_int = op->b.as<ConstIntNode>();
      if (as_int.defined()) {
        if (as_int->value >= 0) {
          Expr beg = SimplifyExpr(op->b * ra->beg);
          Expr extent = SimplifyExpr(op->b * (ra->extent - const_int(1)) + const_int(1));
          Expr step = const_int(1);
          Range bound = Range::make(beg, extent, step);
          this->bounds_[op] = bound;
        } else {
          Expr beg = SimplifyExpr(op->b * (ra->beg + ra->extent - const_int(1)));
          Expr extent = SimplifyExpr((-op->b) * (ra->extent - const_int(1)) + const_int(1));
          Expr step = const_int(1);
          Range bound = Range::make(beg, extent, step);
          this->bounds_[op] = bound;
        }
      } else {
        throw std::runtime_error("InferBound can't handle Range x Range cases.");
      }
    }
  }

  void ImplVisit(FloorDiv op) override {
    ImplVisitBinExpr(op);
    Range ra = this->bounds_[op->a];
    Range rb = this->bounds_[op->b];
    ConstInt as_int = op->b.as<ConstIntNode>();
    if (as_int.defined()) {
      if (as_int->value >= 0) {
        Expr beg = SimplifyExpr(FloorDiv::make(ra->beg, op->b));
        Expr extent = SimplifyExpr(
            FloorDiv::make(ra->extent + (ra->beg % op->b) - const_int(1), op->b) + const_int(1));
        Expr step = const_int(1);
        Range bound = Range::make(beg, extent, step);
        this->bounds_[op] = bound;
      } else {
        Expr beg = SimplifyExpr(FloorDiv::make(ra->beg + ra->extent - const_int(1), op->b));
        Expr extent = SimplifyExpr(
            -FloorDiv::make(ra->extent + ra->beg % op->b - const_int(1), op->b) + const_int(1));
        Expr step = const_int(1);
        Range bound = Range::make(beg, extent, step);
        this->bounds_[op] = bound;
      }
    } else {
      throw std::runtime_error("InferBound can't handle Range//Range cases.");
    }
  }

 public:
  InferExprBound(std::unordered_map<Var, Range> bounds) {
    for (auto kv : bounds) {
      bounds_[kv.first] = kv.second;
    }
  }

  void Visit(IRBase op) override {
    if (bounds_.count(op)) return;
    IRVisitor<>::Visit(op);
  }

 private:
  std::unordered_map<IRBase, Range> bounds_;
};

}  // namespace analysis

}  // namespace domino

#endif  // DOMINO_ANALYSIS_BOUNDS_H