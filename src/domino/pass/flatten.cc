#include <pass/flatten.h>

namespace domino {

namespace pass {

Block FlattenArrayAccess(Block block, std::vector<Var> arrays_to_flatten,
                         std::vector<ExprList> strides) {
  auto expr_mutator = std::make_shared<FlattenNdLoadAndArrayRefMutator>(arrays_to_flatten, strides);
  auto stmt_mutator = std::make_shared<FlattenNdStoreMutator>(arrays_to_flatten, strides, expr_mutator);
  BlockMutator block_mutator(stmt_mutator);
  return block_mutator(block);
}

}  // namespace pass

}  // namespace domino