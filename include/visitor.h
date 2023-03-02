#ifndef DOMINO_VISITOR_H
#define DOMINO_VISITOR_H

#include <ir_functor.h>

namespace domino {

template <typename... Args>
class IRVisitor : public IRFunctor<void(Args...)> {
 protected:
  using IRFunctor<void(Args...)>::Visit;
  void ImplVisit(MemRef op) override {
    Visit(op->var);
    Visit(op->offset);
  }

  void ImplVisit(ValueRef op) override { Visit(op->var); }

  void ImplVisit(ArrayRef op) override {
    Visit(op->var);
    Visit(op->args);
  }

#define X_DECL_BIN_EXPR(OP)        \
  void ImplVisit(OP op) override { \
    Visit(op->a);                  \
    Visit(op->b);                  \
  }
#include <x_macro/bin_expr.x.h>
#undef X_DECL_BIN_EXPR

  void ImplVisit(Cast op) override { Visit(op->a); }

  void ImplVisit(Broadcast op) override { Visit(op->a); }

  void ImplVisit(Neg op) override { Visit(op->a); }

  void ImplVisit(Not op) override { Visit(op->a); }

  void ImplVisit(BitNot op) override { Visit(op->a); }

  void ImplVisit(Ceil op) override { Visit(op->a); }

  void ImplVisit(Floor op) override { Visit(op->a); }

  void ImplVisit(Select op) override {
    Visit(op->a);
    Visit(op->b);
    Visit(op->c);
  }

  void ImplVisit(Range op) override {
    Visit(op->beg);
    Visit(op->extent);
    Visit(op->step);
  }

  void ImplVisit(ExprList op) override {
    for (auto v : op->value_list) {
      Visit(v);
    }
  }

  void ImplVisit(CondAll op) override { Visit(op->phases); }

  void ImplVisit(CondAny op) override { Visit(op->phases); }

  void ImplVisit(Var op) override { return; }

  void ImplVisit(ConstVar op) override { return; }

  void ImplVisit(ConstInt op) override { return; }
  void ImplVisit(ConstUInt op) override { return; }
  void ImplVisit(ConstFloat op) override { return; }
  void ImplVisit(ConstBFloat op) override { return; }
  void ImplVisit(ConstTFloat op) override { return; }
  void ImplVisit(ConstString op) override { return; }

  void ImplVisit(Iterator op) override {
    Visit(op->var);
    Visit(op->range);
  }

  void ImplVisit(NdLoad op) override {
    Visit(op->mem_ref);
    Visit(op->indices);
  }

  void ImplVisit(Load op) override {
    Visit(op->mem_ref);
    Visit(op->addr);
  }

  void ImplVisit(MapVar op) override {
    Visit(op->var);
    Visit(op->expr);
  }

  void ImplVisit(Slice op) override {
    for (auto v : op->indices) {
      Visit(v);
    }
  }

  void ImplVisit(MemSlice op) override {
    Visit(op->var);
    Visit(op->slice);
  }

  void ImplVisit(Call op) override {
    Visit(op->func);
    Visit(op->args);
  }

  void ImplVisit(PackValue op) override { Visit(op->value_list); }

  void ImplVisit(NdStore op) override {
    Visit(op->mem_ref);
    Visit(op->indices);
    Visit(op->value);
  }

  void ImplVisit(Store op) override {
    Visit(op->mem_ref);
    Visit(op->addr);
  }

  void ImplVisit(Evaluate op) override { Visit(op->expr); }

  void ImplVisit(AttrBlock op) override {
    Visit(op->key);
    Visit(op->obj);
    Visit(op->value);
    Visit(op->body);
  }

  void ImplVisit(NdForBlock op) override {
    for (auto it : op->iters) {
      Visit(it);
    }
    Visit(op->compute_level);
    Visit(op->body);
  }

  void ImplVisit(ForBlock op) override {
    Visit(op->iter);
    Visit(op->compute_level);
    Visit(op->body);
  }

  void ImplVisit(BranchBlock op) override {
    Visit(op->cond);
    Visit(op->true_branch);
    Visit(op->false_branch);
  }

  void ImplVisit(SeqBlock op) override {
    Visit(op->first);
    Visit(op->second);
  }

  void ImplVisit(SpatialBlock op) override {
    for (auto block : op->blocks) {
      Visit(block);
    }
    for (auto bind : op->spatial_bindings) {
      Visit(bind);
    }
  }

  void ImplVisit(AtomBlock op) override { Visit(op->getStmt()); }

  void ImplVisit(ReMapBlock op) override {
    for (auto m : op->mappings) {
      Visit(m);
    }
    Visit(op->body);
  }

  void ImplVisit(NdAllocBlock op) override {
    Visit(op->var);
    for (auto s : op->shape) {
      Visit(s);
    }
    Visit(op->memory_scope);
    Visit(op->body);
  }

  void ImplVisit(AllocBlock op) override {
    Visit(op->var);
    Visit(op->memory_scope);
    Visit(op->length);
    Visit(op->body);
  }

  void ImplVisit(ComputeLevel op) override {
    Visit(op->compute_level);
    Visit(op->block);
    for (auto sub : op->sub_levels) {
      Visit(sub);
    }
  }

  void ImplVisit(MemoryLevel op) override {
    Visit(op->memory_level);
    Visit(op->block);
    for (auto sub : op->sub_levels) {
      Visit(sub);
    }
  }

  void ImplVisit(KernelSignature op) override { return; }

  void ImplVisit(Kernel op) override {
    Visit(op->signature);
    Visit(op->body);
  }
};

}  // namespace domino

#endif  // DOMINO_VISITOR_H