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

std::vector<Expr> GetInputTensorIndices(IRBase ir, Var tensor_var) {
  GetNdLoadVisitor visitor;
  visitor(ir);
  std::vector<NdLoad> ndloads = visitor.Get();
  std::vector<Expr> ret;
  bool find = false;
  for (auto ndload : ndloads) {
    if (ndload->mem_ref->var == tensor_var) {
      ret = ndload->indices->value_list;
      find = true;
    }
  }
  ASSERT(find) << "Can't find input tensor " << std::string(tensor_var) << "\n";
  return ret;
}

}  // namespace pass

}  // namespace domino