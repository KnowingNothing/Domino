#include <fmt/core.h>
#include <fmt/ranges.h>
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

class TensorAccessGetter : public IRVisitor<> {
 protected:
  using IRVisitor<>::ImplVisit;

  void ImplVisit(NdStore op) override {
    output_access_[op->mem_ref->var].push_back(op->indices);
    IRVisitor<>::ImplVisit(op);
  }

  void ImplVisit(NdLoad op) override {
    input_access_[op->mem_ref->var].push_back(op->indices);
    IRVisitor<>::ImplVisit(op);
  }

 public:
  using IRVisitor<>::Visit;

  void CheckAndProduceSimpleVisitPattern() {
    for (auto [k, v] : input_access_) {
      ASSERT(v.size() == 1U) << "Can't support access the same tensor with different patterns.\n";
      if (output_access_.count(k)) continue;
      for (auto vv : v[0]->value_list) {
        auto as_var = vv.as<VarNode>();
        auto as_add = vv.as<AddNode>();
        ASSERT(as_var.defined() || as_add.defined())
            << "Only support single var or a simple add expression in index.\n";
        if (as_add.defined()) {
          auto left = as_add->a.as<VarNode>();
          auto right = as_add->b.as<VarNode>();
          ASSERT(left.defined() && right.defined())
              << "For add expression in index, only support Var + Var.\n";
          std::vector<Var> ind;
          ind.push_back(left);
          ind.push_back(right);
          simplified_access_[k].push_back(ind);
          record_dimension(left);
          record_dimension(right);
        } else {
          std::vector<Var> ind;
          ind.push_back(as_var);
          simplified_access_[k].push_back(ind);
          record_dimension(as_var);
        }
      }
    }
    for (auto [k, v] : output_access_) {
      ASSERT(v.size() == 1U) << "Can't support access the same tensor with different patterns.\n";
      for (auto vv : v[0]->value_list) {
        auto as_var = vv.as<VarNode>();
        auto as_add = vv.as<AddNode>();
        ASSERT(as_var.defined() || as_add.defined())
            << "Only support single var or a simple add expression in index.\n";
        if (as_add.defined()) {
          auto left = as_add->a.as<VarNode>();
          auto right = as_add->b.as<VarNode>();
          ASSERT(left.defined() && right.defined())
              << "For add expression in index, only support Var + Var.\n";
          std::vector<Var> ind;
          ind.push_back(left);
          ind.push_back(right);
          simplified_access_[k].push_back(ind);
          record_dimension(left);
          record_dimension(right);
        } else {
          std::vector<Var> ind;
          ind.push_back(as_var);
          simplified_access_[k].push_back(ind);
          record_dimension(as_var);
        }
      }
    }
  }

  std::vector<Var> GetInputVars() {
    std::vector<Var> ret;
    for (auto kv : input_access_) {
      if (!output_access_.count(kv.first)) {
        ret.push_back(kv.first);
      }
    }
    return ret;
  }

  std::vector<Var> GetOutputVars() {
    std::vector<Var> ret;
    for (auto kv : output_access_) {
      ret.push_back(kv.first);
    }
    return ret;
  }

  std::vector<std::vector<Var>> GetSimplifiedAccess(Var v) {
    ASSERT(simplified_access_.count(v)) << "Can't find Var " << std::string(v) << "\n";
    return simplified_access_.at(v);
  }

  std::vector<Var> GetAllDimensions() { return dimensions_; }

 private:
  void record_dimension(Var v) {
    if (!visit_.count(v)) {
      dimensions_.push_back(v);
      visit_[v] += 1;
    }
  }
  std::unordered_map<Var, std::vector<ExprList>> input_access_;
  std::unordered_map<Var, std::vector<ExprList>> output_access_;
  std::unordered_map<Var, std::vector<std::vector<Var>>> simplified_access_;
  std::vector<Var> dimensions_;
  std::unordered_map<Var, int> visit_;
};

std::string GenerateTileFlowOp(IRBase ir) {
  TensorAccessGetter getter;
  getter.Visit(ir);
  getter.CheckAndProduceSimpleVisitPattern();
  std::vector<Var> inputs = getter.GetInputVars();
  std::vector<Var> outputs = getter.GetOutputVars();
  ASSERT(outputs.size() == 1U) << "Only support one output for each operator.\n";

  std::string ret = "  - name: Produce" + outputs[0]->id->value + "\n";
  ret += "    dimensions: [";
  std::vector<std::string> all_dims;
  for (auto d : getter.GetAllDimensions()) {
    all_dims.push_back(d->id->value);
  }
  ret += fmt::format("{}", fmt::join(all_dims, ","));
  ret += "]\n";
  ret += "    data-spaces:\n";
  for (auto out : outputs) {
    std::vector<std::vector<Var>> index = getter.GetSimplifiedAccess(out);
    ret += "    - name: ";
    ret += out->id->value;
    ret += "\n";
    ret += "      projection:\n";
    for (auto ind : index) {
      ret += "        - [";
      std::vector<std::string> tmp;
      for (auto i : ind) {
        tmp.push_back(fmt::format("[{}]", i->id->value));
      }
      ret += fmt::format("{}", fmt::join(tmp, ","));
      ret += "]\n";
    }
    ret += "      read-write: True\n";
  }
  for (auto in : inputs) {
    std::vector<std::vector<Var>> index = getter.GetSimplifiedAccess(in);
    ret += "    - name: ";
    ret += in->id->value;
    ret += "\n";
    ret += "      projection:\n";
    for (auto ind : index) {
      ret += "        - [";
      std::vector<std::string> tmp;
      for (auto i : ind) {
        tmp.push_back(fmt::format("[{}]", i->id->value));
      }
      ret += fmt::format("{}", fmt::join(tmp, ","));
      ret += "]\n";
    }
  }
  ret += "    ins: ";
  std::vector<std::string> input_names;
  for (auto inp : inputs) {
    input_names.push_back(inp->id->value);
  }
  ret += fmt::format("{}", fmt::join(input_names, ", "));
  ret += "\n";
  ret += "    out: ";
  std::vector<std::string> output_names;
  for (auto out : outputs) {
    output_names.push_back(out->id->value);
  }
  ret += fmt::format("{}", fmt::join(output_names, ", "));
  ret += "\n";
  return ret;
}

}  // namespace pass

}  // namespace domino