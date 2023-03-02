#include <ir_base.h>
#include <mutator.h>

namespace domino {

IRBase replicate(IRBase ir) {
  IRMutator mutator;
  return mutator(ir);
}

}  // namespace domino