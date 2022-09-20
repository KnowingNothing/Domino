#ifndef DOMINO_BLOCK_H
#define DOMINO_BLOCK_H

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

class NdForBlockNode : public BlockNode {
 public:
  std::vector<Iterator> iters;
  std::vector<Range> ranges;
  Block body;

  NdForBlockNode(std::vector<Iterator> iters, std::vector<Range> ranges, Block body)
      : iters(std::move(iters)), ranges(std::move(ranges)), body(std::move(body)) {}
};

class BranchBlockNode : public BlockNode {};

class SeqBlockNode : public BlockNode {};

class AtomBlockNode : public BlockNode {};

}  // namespace domino

#endif  // DOMINO_BLOCK_H