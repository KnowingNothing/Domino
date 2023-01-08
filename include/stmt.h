#ifndef DOMINO_STMT_H
#define DOMINO_STMT_H

#include <expr.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <ir_base.h>
#include <logging/logging.h>
#include <ref.h>

namespace domino {
using namespace logging;

// #define X_DECL_STMT(X) \
//   class X##Node;       \
//   using X = Ref<X##Node>;
// #include <x_macro/stmt.x.h>

class StmtNode : public IRBaseNode {
 public:
  virtual operator std::string() const { return fmt::format("Stmt()"); }
};

using Stmt = Ref<StmtNode>;

class NdStoreNode : public StmtNode {
 public:
  NdStoreNode(MemRef mem_ref, ExprList indices, Expr value)
      : mem_ref(std::move(mem_ref)), indices(std::move(indices)), value(std::move(value)) {
    ASSERT(this->mem_ref.defined());
    ASSERT(this->indices.defined());
    ASSERT(this->value.defined());
  }

  operator std::string() const {
    return fmt::format("NdStore({}, {}, {})", std::string(this->mem_ref),
                       std::string(this->indices), std::string(this->value));
  }

  MemRef mem_ref;
  ExprList indices;
  Expr value;
};

using NdStore = Ref<NdStoreNode>;

class StoreNode : public StmtNode {
 public:
  StoreNode(MemRef mem_ref, Expr addr, Expr value)
      : mem_ref(std::move(mem_ref)), addr(std::move(addr)), value(std::move(value)) {
    ASSERT(this->mem_ref.defined());
    ASSERT(this->addr.defined());
    ASSERT(this->value.defined());
  }

  operator std::string() const {
    return fmt::format("Store({}, {}, {})", std::string(this->mem_ref), std::string(this->addr),
                       std::string(this->value));
  }

  MemRef mem_ref;
  Expr addr;
  Expr value;
};

using Store = Ref<StoreNode>;

class EvaluateNode : public StmtNode {
 public:
  EvaluateNode(Expr expr) : expr(std::move(expr)) {}

  operator std::string() const { return fmt::format("Evaluate({})", std::string(this->expr)); }

  Expr expr;
};

using Evaluate = Ref<EvaluateNode>;

}  // namespace domino

#endif  // DOMINO_STMT_H
