#include <pass/prod_consum.h>

namespace domino {

namespace pass {

std::vector<Var> GetInputTensorVars(IRBase ir) {
  GetNdLoadVisitor visitor;
  visitor(ir);
  std::vector<NdLoad> ndloads = visitor.Get();
  std::vector<Var> ret;
  for (auto nd : ndloads) {
    ret.push_back(nd->mem_ref->var);
  }
  return ret;
}

}  // namespace pass

}  // namespace domino