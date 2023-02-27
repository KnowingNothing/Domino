#ifndef DOMINO_BLOCK_H
#define DOMINO_BLOCK_H

// #include <arch.h>
#include <expr.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <ir_base.h>
#include <ref.h>
#include <stmt.h>

#include <vector>

namespace domino {

/// Don't use X_Macro for reference declaration
/// for better debug experience

// #define X_DECL_BLOCK(X) \
//   class X##Node;        \
//   using X = Ref<X##Node>;
// #include <x_macro/block.x.h>

class BlockNode : public IRBaseNode {};

using Block = Ref<BlockNode>;

class AttrBlockNode : public BlockNode {
 public:
  AttrBlockNode(std::string key, Var obj, Expr value, Block body)
      : key(ConstString::make(std::move(key))),
        obj(std::move(obj)),
        value(std::move(value)),
        body(std::move(body)) {
    ASSERT(this->key.defined());
    ASSERT(this->obj.defined());
    ASSERT(this->value.defined());
    ASSERT(this->body.defined());
  }

  AttrBlockNode(ConstString key, Var obj, Expr value, Block body)
      : key(std::move(key)), obj(std::move(obj)), value(std::move(value)), body(std::move(body)) {
    ASSERT(this->key.defined());
    ASSERT(this->obj.defined());
    ASSERT(this->value.defined());
    ASSERT(this->body.defined());
  }

  ConstString key;
  Var obj;
  Expr value;
  Block body;
};

using AttrBlock = Ref<AttrBlockNode>;

class NdForBlockNode : public BlockNode {
 public:
  NdForBlockNode(std::vector<Iterator> iters, Block body,
                 ConstString compute_level = ConstString::make(""))
      : iters(std::move(iters)), body(std::move(body)), compute_level(std::move(compute_level)) {
    for (auto i : this->iters) {
      ASSERT(i.defined());
    }
    ASSERT(this->body.defined());
  }

  NdForBlockNode(std::vector<Iterator> iters, Block body,
                 std::string compute_level = std::string(""))
      : iters(std::move(iters)),
        body(std::move(body)),
        compute_level(ConstString::make(std::move(compute_level))) {
    for (auto i : this->iters) {
      ASSERT(i.defined());
    }
    ASSERT(this->body.defined());
  }

  std::vector<Iterator> iters;
  Block body;
  ConstString compute_level;
};

using NdForBlock = Ref<NdForBlockNode>;

class ForBlockNode : public BlockNode {
 public:
  ForBlockNode(Iterator iter, Block body, ConstString compute_level = ConstString::make(""))
      : iter(std::move(iter)), body(std::move(body)), compute_level(std::move(compute_level)) {
    ASSERT(this->iter.defined());
    ASSERT(this->body.defined());
    ASSERT(this->compute_level.defined());
  }

  ForBlockNode(Iterator iter, Block body, std::string compute_level = std::string(""))
      : iter(std::move(iter)),
        body(std::move(body)),
        compute_level(ConstString::make(std::move(compute_level))) {
    ASSERT(this->iter.defined());
    ASSERT(this->body.defined());
    ASSERT(this->compute_level.defined());
  }

  Iterator iter;
  Block body;
  ConstString compute_level;
};

using ForBlock = Ref<ForBlockNode>;

class BranchBlockNode : public BlockNode {
 public:
  BranchBlockNode(Expr cond, Block true_branch, Block false_branch)
      : cond(std::move(cond)),
        true_branch(std::move(true_branch)),
        false_branch(std::move(false_branch)) {
    ASSERT(this->cond.defined());
    ASSERT(this->true_branch.defined());
    ASSERT(this->false_branch.defined());
  }

  Expr cond;
  Block true_branch;
  Block false_branch;
};

using BranchBlock = Ref<BranchBlockNode>;

class SeqBlockNode : public BlockNode {
 public:
  SeqBlockNode(Block first, Block second) : first(std::move(first)), second(std::move(second)) {
    ASSERT(this->first.defined());
    ASSERT(this->second.defined());
  }

  Block first;
  Block second;
};

using SeqBlock = Ref<SeqBlockNode>;

class SpatialBlockNode : public BlockNode {
 public:
  SpatialBlockNode(std::vector<Block> blocks, std::vector<ConstString> spatial_bindings)
      : blocks(std::move(blocks)), spatial_bindings(std::move(spatial_bindings)) {
    for (auto b : this->blocks) {
      ASSERT(b.defined());
    }
    for (auto s : this->spatial_bindings) {
      ASSERT(s.defined());
    }

    ASSERT(this->blocks.size() == this->spatial_bindings.size());
  }

  std::vector<Block> blocks;
  std::vector<ConstString> spatial_bindings;
};

using SpatialBlock = Ref<SpatialBlockNode>;

class AtomBlockNode : public BlockNode {
 public:
  AtomBlockNode() : stmt(nullptr) {}
  AtomBlockNode(Stmt stmt) : stmt(std::move(stmt)) { ASSERT(this->stmt.defined()); }

  bool isNullBlock() const { return !stmt.defined(); }

  Stmt getStmt() const {
    ASSERT(!this->isNullBlock()) << "Can't get statement from empty block.";
    return stmt;
  }

  static AtomBlockNode makeNullBlock() { return AtomBlockNode(); }

 private:
  Stmt stmt;
};

using AtomBlock = Ref<AtomBlockNode>;

class ReMapBlockNode : public BlockNode {
 public:
  ReMapBlockNode(std::vector<MapVar> mappings, Block body)
      : mappings(std::move(mappings)), body(body) {
    for (auto m : this->mappings) {
      ASSERT(m.defined());
    }
    ASSERT(this->body.defined());
  }

  std::vector<MapVar> mappings;
  Block body;
};

using ReMapBlock = Ref<ReMapBlockNode>;

class NdAllocBlockNode : public BlockNode {
 public:
  NdAllocBlockNode(Var var, std::vector<Expr> shape, ConstString memory_scope, Block body)
      : var(std::move(var)),
        shape(std::move(shape)),
        memory_scope(std::move(memory_scope)),
        body(std::move(body)) {
    ASSERT(this->var.defined());
    for (auto s : this->shape) {
      ASSERT(s.defined());
    }
    ASSERT(this->memory_scope.defined());
    ASSERT(this->body.defined());
  }

  Var var;
  std::vector<Expr> shape;
  ConstString memory_scope;
  Block body;
};

using NdAllocBlock = Ref<NdAllocBlockNode>;

class AllocBlockNode : public BlockNode {
 public:
  AllocBlockNode(Var var, Expr length, ConstString memory_scope, Block body)
      : var(std::move(var)),
        length(std::move(length)),
        memory_scope(std::move(memory_scope)),
        body(std::move(body)) {
    ASSERT(this->var.defined());
    ASSERT(this->length.defined());
    ASSERT(this->memory_scope.defined());
    ASSERT(this->body.defined());
  }

  Var var;
  Expr length;
  ConstString memory_scope;
  Block body;
};

using AllocBlock = Ref<AllocBlockNode>;

}  // namespace domino

#endif  // DOMINO_BLOCK_H