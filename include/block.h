#ifndef DOMINO_BLOCK_H
#define DOMINO_BLOCK_H

#include <arch.h>
#include <expr.h>
#include <ir_base.h>
#include <ref.h>
#include <stmt.h>

#include <vector>

namespace domino {

#define X_DECL_BLOCK(X) \
  class X##Node;        \
  using X = Ref<X##Node>;
#include <x_macro/block.x.h>

class BlockNode : public IRBaseNode {};

class AttrBlockNode : public BlockNode {
 public:
  std::string key;
  IRBase obj;
  IRBase value;
  Block body;

  AttrBlockNode(std::string key, IRBase obj, IRBase value, Block body)
      : key(std::move(key)), obj(std::move(obj)), value(std::move(value)), body(std::move(body)) {}
};

class NdForBlockNode : public BlockNode {
 public:
  std::vector<Iterator> iters;
  std::vector<Range> ranges;
  Block body;
  arch::CompLevel level;

  NdForBlockNode(std::vector<Iterator> iters, std::vector<Range> ranges, Block body,
                 arch::CompLevel level = arch::CompLevel::dUNKNOWN)
      : iters(std::move(iters)), ranges(std::move(ranges)), body(std::move(body)), level(level) {}
};

class BranchBlockNode : public BlockNode {
 public:
  std::vector<Expr> conds;
  Block true_branch;
  Block false_branch;

  BranchBlockNode(std::vector<Expr> conds, Block true_branch, Block false_branch)
      : conds(std::move(conds)),
        true_branch(std::move(true_branch)),
        false_branch(std::move(false_branch)) {}
};

class SeqBlockNode : public BlockNode {
 public:
  Block first;
  Block second;

  SeqBlockNode(Block first, Block second) : first(std::move(first)), second(std::move(second)) {}
};

class AtomBlockNode : public BlockNode {
 public:
  Stmt stmt;

  AtomBlockNode(Stmt stmt) : stmt(std::move(stmt)) {}
};

class ReMapBlockNode : public BlockNode {
 public:
  std::vector<Var> vars;     // the produced vars after mapping
  std::vector<Expr> ftrans;  // the forward transformation
  std::vector<Expr> btrans;  // the backward transformation, optionally null
  Block body;

  ReMapBlockNode(std::vector<Var> vars, std::vector<Expr> ftrans, std::vector<Expr> btrans)
      : vars(std::move(vars)), ftrans(std::move(ftrans)), btrans(std::move(btrans)) {}
};

class AllocBlockNode : public BlockNode {
 public:
  Var var;
  std::vector<Expr> shape;
  arch::MemoryScope scope;
  Block body;

  AllocBlockNode(Var var, std::vector<Expr> shape, arch::MemoryScope scope, Block body)
      : var(std::move(var)),
        shape(std::move(shape)),
        scope(std::move(scope)),
        body(std::move(body)) {}
};

}  // namespace domino

#endif  // DOMINO_BLOCK_H