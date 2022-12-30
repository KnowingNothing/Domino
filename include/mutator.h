#ifndef DOMINO_MUTATOR_H
#define DOMINO_MUTATOR_H

#include <expr.h>
#include <ir_functor.h>

namespace domino {

class ExprMutator : public IRFunctor<Expr()> {
 protected:
  Expr ImplVisit(MemRef op) override {
    Expr var = Visit(op->var);
    Var as_var = var.as<VarNode>();
    ASSERT(as_var.defined());
    return MemRef::make(as_var, Visit(op->offset));
  }

#define VISIT_BIN(OP) \
  Expr ImplVisit(OP op) override { return OP::make(Visit(op->a), Visit(op->b)); }

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

  Expr ImplVisit(Cast op) override { return Cast::make(op->dtype, Visit(op->a)); }

  Expr ImplVisit(Broadcast op) override { return Broadcast::make(op->dtype, Visit(op->a)); }

  Expr ImplVisit(Neg op) override { return Neg::make(Visit(op->a)); }

  Expr ImplVisit(Not op) override { return Not::make(Visit(op->a)); }

  Expr ImplVisit(BitNot op) override { return BitNot::make(Visit(op->a)); }

  Expr ImplVisit(Ceil op) override { return Ceil::make(op->dtype, Visit(op->a)); }

  Expr ImplVisit(Floor op) override { return Floor::make(op->dtype, Visit(op->a)); }

  Expr ImplVisit(Select op) override {
    return Select::make(Visit(op->a), Visit(op->b), Visit(op->c));
  }

  Expr ImplVisit(Range op) override {
    return Range::make(Visit(op->beg), Visit(op->extent), Visit(op->step));
  }

  Expr ImplVisit(ExprList op) override {
    int length = (int)op->value_list.size();
    std::vector<Expr> value_list;
    for (int i = 0; i < length; ++i) {
      value_list.push_back(Visit(op->value_list[i]));
    }
    return ExprList::make(value_list);
  }

  Expr ImplVisit(CondAll op) override {
    Expr list = Visit(op->phases);
    ExprList as_list = list.as<ExprListNode>();
    ASSERT(as_list.defined());
    return CondAll::make(as_list);
  }

  Expr ImplVisit(CondAny op) override {
    Expr list = Visit(op->phases);
    ExprList as_list = list.as<ExprListNode>();
    ASSERT(as_list.defined());
    return CondAny::make(as_list);
  }

  Expr ImplVisit(ConstInt op) override {
    return ConstInt::make(op->value, op->dtype.bits, op->dtype.lane);
  }

  Expr ImplVisit(ConstUInt op) override {
    return ConstUInt::make(op->value, op->dtype.bits, op->dtype.lane);
  }

  Expr ImplVisit(ConstFloat op) override {
    return ConstFloat::make(op->value, op->dtype.bits, op->dtype.lane);
  }

  Expr ImplVisit(ConstBFloat op) override {
    return ConstBFloat::make(op->value, op->dtype.bits, op->dtype.lane);
  }

  Expr ImplVisit(ConstTFloat op) override {
    return ConstTFloat::make(op->value, op->dtype.bits, op->dtype.lane);
  }

  Expr ImplVisit(ConstString op) override { return ConstString::make(op->value); }

  Expr ImplVisit(Var op) override {
    Expr id = Visit(op->id);
    ConstString as_id = id.as<ConstStringNode>();
    ASSERT(as_id.defined());
    return Var::make(op->dtype, as_id);
  }

  Expr ImplVisit(Iterator op) override {
    Expr var = Visit(op->var);
    Var as_var = var.as<VarNode>();
    ASSERT(as_var.defined());
    Expr range = Visit(op->range);
    Range as_range = range.as<RangeNode>();
    ASSERT(as_range.defined());
    return Iterator::make(as_var, as_range, op->iter_type);
  }

  Expr ImplVisit(NdLoad op) override {
    Expr mem_ref = Visit(op->mem_ref);
    MemRef as_mem_ref = mem_ref.as<MemRefNode>();
    ASSERT(as_mem_ref.defined());
    Expr list = Visit(op->indices);
    ExprList as_list = list.as<ExprListNode>();
    ASSERT(as_list.defined());
    return NdLoad::make(as_mem_ref, as_list);
  }

  Expr ImplVisit(Load op) override {
    Expr mem_ref = Visit(op->mem_ref);
    MemRef as_mem_ref = mem_ref.as<MemRefNode>();
    ASSERT(as_mem_ref.defined());
    Expr addr = Visit(op->addr);
    return Load::make(as_mem_ref, addr);
  }

  Expr ImplVisit(MapVar op) override {
    Expr var = Visit(op->var);
    Var as_var = var.as<VarNode>();
    ASSERT(as_var.defined());
    return MapVar::make(as_var, Visit(op->expr));
  }

  Expr ImplVisit(Slice op) override {
    int length = (int)op->indices.size();
    std::vector<Range> indices;
    for (int i = 0; i < length; ++i) {
      Expr r = Visit(op->indices[i]);
      Range as_r = r.as<RangeNode>();
      ASSERT(as_r.defined());
      indices.push_back(as_r);
    }
    return Slice::make(indices);
  }

  Expr ImplVisit(MemSlice op) override {
    Expr var = Visit(op->var);
    Var as_var = var.as<VarNode>();
    ASSERT(as_var.defined());
    Expr slice = Visit(op->slice);
    Slice as_slice = var.as<SliceNode>();
    ASSERT(as_slice.defined());
    return MemSlice::make(as_var, Visit(op->offset), as_slice);
  }

  Expr ImplVisit(Call op) override {
    Expr func = Visit(op->func);
    ConstString as_str = func.as<ConstStringNode>();
    ASSERT(as_str.defined());
    Expr list = Visit(op->args);
    ExprList as_list = list.as<ExprListNode>();
    ASSERT(as_list.defined());
    return Call::make(op->dtype, as_str, as_list);
  }
};

}  // namespace domino

#endif  // DOMINO_MUTATOR_H