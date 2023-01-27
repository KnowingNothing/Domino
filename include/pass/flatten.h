#ifndef DOMINO_PASS_FLATTEN_H
#define DOMINO_PASS_FLATTEN_H

#include <mutator.h>

namespace domino {

namespace pass {

class FlattenNdLoadAndArrayRefMutator : public ExprMutator {
 protected:
  using ExprMutator::Visit;
  Expr ImplVisit(NdLoad op) override {
    Expr mem_ref = Visit(op->mem_ref);
    MemRef as_mem_ref = mem_ref.as<MemRefNode>();
    ASSERT(as_mem_ref.defined());
    Expr list = Visit(op->indices);
    ExprList as_list = list.as<ExprListNode>();
    ASSERT(as_list.defined());
    if (table_.count(as_mem_ref->var)) {
      ExprList strides = table_.at(as_mem_ref->var);
      int len = (int)strides->size();
      ASSERT(len == (int)as_list->size())
          << "Strides should has the same length as indices.\n"
          << std::string(strides) << " vs " << std::string(as_list) << "\n";
      if (len > 0) {
        Expr flattened = as_list->value_list[0];
        // skip the first stride as it is not used
        for (int i = 1; i < len; ++i) {
          flattened = (flattened * strides->value_list[i] + as_list->value_list[i]);
        }
        return Load::make(as_mem_ref, flattened);
      }
    }
    return NdLoad::make(as_mem_ref, as_list);
  }

  Expr ImplVisit(ArrayRef op) override {
    Expr var = Visit(op->var);
    Var as_var = var.as<VarNode>();
    ASSERT(as_var.defined());
    Expr list = Visit(op->args);
    ExprList as_list = list.as<ExprListNode>();
    ASSERT(as_list.defined());

    if (table_.count(as_var)) {
      ExprList strides = table_.at(as_var);
      int len = (int)strides->size();
      ASSERT(len == (int)as_list->size())
          << "Strides should has the same length as indices." << std::string(strides) << " vs "
          << std::string(as_list) << "\n";
      if (len > 0) {
        Expr flattened = as_list->value_list[0];
        // skip the first stride as it is not used
        for (int i = 1; i < len; ++i) {
          flattened = (flattened * strides->value_list[i] + as_list->value_list[i]);
        }
        return MemRef::make(as_var, flattened);
      }
    }
    return ArrayRef::make(as_var, as_list);
  }

 public:
  FlattenNdLoadAndArrayRefMutator(std::vector<Var> arrays_to_flatten,
                                  std::vector<ExprList> strides) {
    ASSERT(arrays_to_flatten.size() == strides.size())
        << "Given arrays and strides length mismatch.";
    int length = (int)arrays_to_flatten.size();
    for (int i = 0; i < length; ++i) {
      table_[arrays_to_flatten[i]] = strides[i];
    }
  }

 private:
  std::unordered_map<Var, ExprList> table_;
};

class FlattenNdStoreMutator : public StmtMutator {
 protected:
  using StmtMutator::Visit;
  Stmt ImplVisit(NdStore op) override {
    Expr mem_ref = VisitExpr(op->mem_ref);
    MemRef as_ref = mem_ref.as<MemRefNode>();
    ASSERT(as_ref.defined());
    Expr list = VisitExpr(op->indices);
    ExprList as_list = list.as<ExprListNode>();
    ASSERT(as_list.defined());

    if (table_.count(as_ref->var)) {
      ExprList strides = table_.at(as_ref->var);
      int len = (int)strides->size();
      ASSERT(len == (int)as_list->size()) << "Strides should has the same length as indices.";
      if (len > 0) {
        Expr flattened = as_list->value_list[0];
        // skip the first stride as it is not used
        for (int i = 1; i < len; ++i) {
          flattened = (flattened * strides->value_list[i] + as_list->value_list[i]);
        }
        return Store::make(as_ref, flattened, VisitExpr(op->value));
      }
    }
    return NdStore::make(as_ref, as_list, VisitExpr(op->value));
  }

 public:
  FlattenNdStoreMutator(std::vector<Var> arrays_to_flatten, std::vector<ExprList> strides)
      : StmtMutator() {
    ASSERT(arrays_to_flatten.size() == strides.size())
        << "Given arrays and strides length mismatch.";
    int length = (int)arrays_to_flatten.size();
    for (int i = 0; i < length; ++i) {
      table_[arrays_to_flatten[i]] = strides[i];
    }
  }
  FlattenNdStoreMutator(std::vector<Var> arrays_to_flatten, std::vector<ExprList> strides,
                        std::shared_ptr<ExprMutator> expr_mutator)
      : StmtMutator(expr_mutator) {
    ASSERT(arrays_to_flatten.size() == strides.size())
        << "Given arrays and strides length mismatch.";
    int length = (int)arrays_to_flatten.size();
    for (int i = 0; i < length; ++i) {
      table_[arrays_to_flatten[i]] = strides[i];
    }
  }

 private:
  std::unordered_map<Var, ExprList> table_;
};

Block FlattenArrayAccess(Block block, std::vector<Var> arrays_to_flatten,
                         std::vector<ExprList> strides);

}  // namespace pass

}  // namespace domino

#endif  // DOMINO_PASS_FLATTEN_H