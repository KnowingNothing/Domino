#ifndef DOMINO_MUTATOR_H
#define DOMINO_MUTATOR_H

#include <expr.h>
#include <ir_functor.h>

namespace domino {

class ExprMutator : public IRFunctor<Expr()> {
 protected:
  Expr ImplVisit(Expr op) override { return op; }

  Expr ImplVisit(MemRef op) override {
    Expr var = Visit(op->var);
    Var as_var = var.as<VarNode>();
    ASSERT(as_var.defined());
    return MemRef::make(as_var, Visit(op->offset));
  }

  Expr ImplVisit(ValueRef op) override {
    Expr var = Visit(op->var);
    Var as_var = var.as<VarNode>();
    ASSERT(as_var.defined());
    return ValueRef::make(as_var);
  }

  Expr ImplVisit(ArrayRef op) override {
    Expr var = Visit(op->var);
    Var as_var = var.as<VarNode>();
    ASSERT(as_var.defined());
    Expr list = Visit(op->args);
    ExprList as_list = list.as<ExprListNode>();
    ASSERT(as_list.defined());
    return ArrayRef::make(as_var, as_list);
  }

#define X_DECL_BIN_EXPR(OP) \
  Expr ImplVisit(OP op) override { return OP::make(Visit(op->a), Visit(op->b)); }
#include <x_macro/bin_expr.x.h>
#undef X_DECL_BIN_EXPR

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

  Expr ImplVisit(Var op) override {
    return op;
    // Expr id = Visit(op->id);
    // ConstString as_id = id.as<ConstStringNode>();
    // ASSERT(as_id.defined());
    // return Var::make(op->dtype, as_id);
  }

  Expr ImplVisit(ConstVar op) override {
    return op;
    // Expr id = Visit(op->id);
    // ConstString as_id = id.as<ConstStringNode>();
    // ASSERT(as_id.defined());
    // return ConstVar::make(op->dtype, as_id);
  }

  Expr ImplVisit(ConstInt op) override { return op; }

  Expr ImplVisit(ConstUInt op) override { return op; }

  Expr ImplVisit(ConstFloat op) override { return op; }

  Expr ImplVisit(ConstBFloat op) override { return op; }

  Expr ImplVisit(ConstTFloat op) override { return op; }

  Expr ImplVisit(ConstString op) override { return ConstString::make(op->value); }

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

  Expr ImplVisit(PackValue op) override {
    Expr list = Visit(op->value_list);
    ExprList as_list = list.as<ExprListNode>();
    ASSERT(as_list.defined());
    return PackValue::make(op->dtype, as_list);
  }

 public:
  Expr Visit(IRBase op) override {
    if (visit_cache_.count(op)) {
      return visit_cache_.at(op);
    } else {
      Expr ret = IRFunctor<Expr()>::Visit(op);
      visit_cache_[op] = ret;
      return ret;
    }
  }

 private:
  std::unordered_map<IRBase, Expr> visit_cache_;
};

class StmtMutator : public IRFunctor<Stmt()> {
 protected:
  Stmt ImplVisit(Stmt op) override { return op; }

  Stmt ImplVisit(NdStore op) override {
    Expr mem_ref = VisitExpr(op->mem_ref);
    MemRef as_ref = mem_ref.as<MemRefNode>();
    ASSERT(as_ref.defined());
    Expr list = VisitExpr(op->indices);
    ExprList as_list = list.as<ExprListNode>();
    ASSERT(as_list.defined());
    return NdStore::make(as_ref, as_list, VisitExpr(op->value));
  }

  Stmt ImplVisit(Store op) override {
    Expr mem_ref = VisitExpr(op->mem_ref);
    MemRef as_ref = mem_ref.as<MemRefNode>();
    ASSERT(as_ref.defined());
    return Store::make(as_ref, VisitExpr(op->addr), VisitExpr(op->value));
  }

  Stmt ImplVisit(Evaluate op) override { return Evaluate::make(VisitExpr(op->expr)); }

 public:
  StmtMutator() : expr_mutator(std::make_shared<ExprMutator>()) {}
  StmtMutator(std::shared_ptr<ExprMutator> mutator) : expr_mutator(mutator) {}
  Expr VisitExpr(Expr expr) { return expr_mutator->Visit(expr); }

 protected:
  std::shared_ptr<ExprMutator> expr_mutator = (nullptr);

 public:
  Stmt Visit(IRBase op) override {
    if (visit_cache_.count(op)) {
      return visit_cache_.at(op);
    } else {
      Stmt ret = IRFunctor<Stmt()>::Visit(op);
      visit_cache_[op] = ret;
      return ret;
    }
  }

 private:
  std::unordered_map<IRBase, Stmt> visit_cache_;
};

class BlockMutator : public IRFunctor<Block()> {
 protected:
  Block ImplVisit(Block op) override { return op; }

  Block ImplVisit(AttrBlock op) override {
    Expr key = VisitExpr(op->key);
    ConstString as_str = key.as<ConstStringNode>();
    ASSERT(as_str.defined());
    Expr v = VisitExpr(op->obj);
    Var as_var = v.as<VarNode>();
    ASSERT(as_var.defined());
    return AttrBlock::make(as_str, as_var, VisitExpr(op->value), Visit(op->body));
  }

  Block ImplVisit(NdForBlock op) override {
    std::vector<Iterator> iters;
    for (auto it : op->iters) {
      Expr iter = VisitExpr(it);
      Iterator as_iter = iter.as<IteratorNode>();
      ASSERT(as_iter.defined());
      iters.push_back(as_iter);
    }
    Expr level = VisitExpr(op->compute_level);
    ConstString as_str = level.as<ConstStringNode>();
    ASSERT(as_str.defined());
    return NdForBlock::make(iters, Visit(op->body), as_str);
  }

  Block ImplVisit(ForBlock op) override {
    Expr iter = VisitExpr(op->iter);
    Iterator as_iter = iter.as<IteratorNode>();
    ASSERT(as_iter.defined());
    Expr level = VisitExpr(op->compute_level);
    ConstString as_str = level.as<ConstStringNode>();
    ASSERT(as_str.defined());
    return ForBlock::make(as_iter, Visit(op->body), as_str);
  }

  Block ImplVisit(BranchBlock op) override {
    return BranchBlock::make(VisitExpr(op->cond), Visit(op->true_branch), Visit(op->false_branch));
  }

  Block ImplVisit(SeqBlock op) override {
    return SeqBlock::make(Visit(op->first), Visit(op->second));
  }

  Block ImplVisit(SpatialBlock op) override {
    std::vector<Block> blocks;
    for (auto b : op->blocks) {
      blocks.push_back(Visit(b));
    }
    std::vector<ConstString> bindings;
    for (auto s : op->spatial_bindings) {
      Expr binding = VisitExpr(s);
      ConstString as_str = binding.as<ConstStringNode>();
      ASSERT(as_str.defined());
      bindings.push_back(as_str);
    }
    return SpatialBlock::make(blocks, bindings);
  }

  Block ImplVisit(AtomBlock op) override { return AtomBlock::make(VisitStmt(op->getStmt())); }

  Block ImplVisit(ReMapBlock op) override {
    std::vector<MapVar> mappings;
    for (auto m : op->mappings) {
      Expr mapping = VisitExpr(m);
      MapVar as_map = mapping.as<MapVarNode>();
      mappings.push_back(as_map);
    }
    return ReMapBlock::make(mappings, Visit(op->body));
  }

  Block ImplVisit(NdAllocBlock op) override {
    Expr v = VisitExpr(op->var);
    Var as_var = v.as<VarNode>();
    ASSERT(as_var.defined());
    std::vector<Expr> shape;
    for (auto s : op->shape) {
      shape.push_back(VisitExpr(s));
    }
    Expr level = VisitExpr(op->memory_scope);
    ConstString as_str = level.as<ConstStringNode>();
    ASSERT(as_str.defined());
    return NdAllocBlock::make(as_var, shape, as_str, Visit(op->body));
  }

  Block ImplVisit(AllocBlock op) override {
    Expr v = VisitExpr(op->var);
    Var as_var = v.as<VarNode>();
    ASSERT(as_var.defined());
    Expr level = VisitExpr(op->memory_scope);
    ConstString as_str = level.as<ConstStringNode>();
    ASSERT(as_str.defined());
    return AllocBlock::make(as_var, VisitExpr(op->length), as_str, Visit(op->body));
  }

 public:
  BlockMutator() : stmt_mutator(std::make_shared<StmtMutator>()) {}
  BlockMutator(std::shared_ptr<ExprMutator> mutator)
      : stmt_mutator(std::make_shared<StmtMutator>(mutator)) {}
  BlockMutator(std::shared_ptr<StmtMutator> mutator) : stmt_mutator(mutator) {}
  Expr VisitExpr(Expr expr) { return stmt_mutator->VisitExpr(expr); }
  Stmt VisitStmt(Stmt expr) { return stmt_mutator->Visit(expr); }

 protected:
  std::shared_ptr<StmtMutator> stmt_mutator = (nullptr);

 public:
  Block Visit(IRBase op) override {
    if (visit_cache_.count(op)) {
      return visit_cache_.at(op);
    } else {
      Block ret = IRFunctor<Block()>::Visit(op);
      visit_cache_[op] = ret;
      return ret;
    }
  }

 private:
  std::unordered_map<IRBase, Block> visit_cache_;
};

class ArchMutator : public IRFunctor<Arch()> {
 protected:
  Arch ImplVisit(MemoryLevel op) override {
    Expr level = VisitExpr(op->memory_level);
    ConstInt as_int = level.as<ConstIntNode>();
    ASSERT(as_int.defined());
    std::vector<Arch> new_sub;
    for (auto v : op->sub_levels) {
      new_sub.push_back(Visit(v));
    }
    auto ret = MemoryLevel::make(as_int, VisitBlock(op->block), new_sub);
    ret->scope = op->scope;
    return ret;
  }

  Arch ImplVisit(ComputeLevel op) override {
    Expr level = VisitExpr(op->compute_level);
    ConstInt as_int = level.as<ConstIntNode>();
    ASSERT(as_int.defined());
    std::vector<Arch> new_sub;
    for (auto v : op->sub_levels) {
      new_sub.push_back(Visit(v));
    }
    return ComputeLevel::make(as_int, VisitBlock(op->block), new_sub);
  }

 public:
  ArchMutator() : block_mutator(std::make_shared<BlockMutator>()) {}
  ArchMutator(std::shared_ptr<ExprMutator> mutator)
      : block_mutator(std::make_shared<BlockMutator>(mutator)) {}
  ArchMutator(std::shared_ptr<StmtMutator> mutator)
      : block_mutator(std::make_shared<BlockMutator>(mutator)) {}
  ArchMutator(std::shared_ptr<BlockMutator> mutator) : block_mutator(mutator) {}
  Expr VisitExpr(Expr expr) { return block_mutator->VisitExpr(expr); }
  Stmt VisitStmt(Stmt stmt) { return block_mutator->VisitStmt(stmt); }
  Block VisitBlock(Block block) { return block_mutator->Visit(block); }

 protected:
  std::shared_ptr<BlockMutator> block_mutator = (nullptr);

 public:
  Arch Visit(IRBase op) override {
    if (visit_cache_.count(op)) {
      return visit_cache_.at(op);
    } else {
      Arch ret = IRFunctor<Arch()>::Visit(op);
      visit_cache_[op] = ret;
      return ret;
    }
  }

 private:
  std::unordered_map<IRBase, Arch> visit_cache_;
};

class IRMutator : public IRFunctor<IRBase()> {
 protected:
  IRBase ImplVisit(KernelSignature op) override {
    return KernelSignature::make(op->kernel_name, op->kernel_args, op->scalar_args);
  }

  IRBase ImplVisit(Kernel op) override {
    IRBase signature = Visit(op->signature);
    KernelSignature as_sig = signature.as<KernelSignatureNode>();
    ASSERT(as_sig.defined());
    Kernel ret = Kernel::make(as_sig, VisitBlock(op->body));
    /// FIXME: ugly source copy
    ret->source = op->source;
    return ret;
  }

 public:
  IRMutator() : arch_mutator(std::make_shared<ArchMutator>()) {}
  IRMutator(std::shared_ptr<ExprMutator> mutator)
      : arch_mutator(std::make_shared<ArchMutator>(mutator)) {}
  IRMutator(std::shared_ptr<StmtMutator> mutator)
      : arch_mutator(std::make_shared<ArchMutator>(mutator)) {}
  IRMutator(std::shared_ptr<BlockMutator> mutator)
      : arch_mutator(std::make_shared<ArchMutator>(mutator)) {}
  IRMutator(std::shared_ptr<ArchMutator> mutator) : arch_mutator(mutator) {}
  Expr VisitExpr(Expr expr) { return arch_mutator->VisitExpr(expr); }
  Stmt VisitStmt(Stmt stmt) { return arch_mutator->VisitStmt(stmt); }
  Block VisitBlock(Block block) { return arch_mutator->VisitBlock(block); }
  Arch VisitArch(Arch arch) { return arch_mutator->Visit(arch); }

 protected:
  std::shared_ptr<ArchMutator> arch_mutator = (nullptr);

 public:
  IRBase Visit(IRBase op) override {
    if (visit_cache_.count(op)) {
      return visit_cache_.at(op);
    } else {
      IRBase ret;
      if (op.as<ExprNode>().defined()) {
        ret = VisitExpr(op.as<ExprNode>());
      } else if (op.as<StmtNode>().defined()) {
        ret = VisitStmt(op.as<StmtNode>());
      } else if (op.as<BlockNode>().defined()) {
        ret = VisitBlock(op.as<BlockNode>());
      } else if (op.as<ArchNode>().defined()) {
        ret = VisitArch(op.as<ArchNode>());
      } else {
        ret = IRFunctor<IRBase()>::Visit(op);
      }
      visit_cache_[op] = ret;
      return ret;
    }
  }

 private:
  std::unordered_map<IRBase, IRBase> visit_cache_;
};

}  // namespace domino

#endif  // DOMINO_MUTATOR_H