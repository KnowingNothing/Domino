#ifndef DOMINO_PASS_PROD_CONSUM_H
#define DOMINO_PASS_PROD_CONSUM_H

#include <visitor.h>

#include <vector>

namespace domino {

namespace pass {

class GetNdLoadVisitor : public IRVisitor<> {
 protected:
  void ImplVisit(NdLoad op) override { ndloads_.push_back(op); }

 public:
  std::vector<NdLoad> Get() { return ndloads_; }

 private:
  std::vector<NdLoad> ndloads_;
};

std::vector<Var> GetInputTensorVars(IRBase ir);

std::vector<Expr> GetInputTensorIndices(IRBase ir, Var tensor_var);

}  // namespace pass

}  // namespace domino

#endif  // DOMINO_PASS_PROD_CONSUM_H