#ifndef DOMINO_STMT_H
#define DOMINO_STMT_H

#include <ir_base.h>
#include <ref.h>

namespace domino {

#define X_DECL_STMT(X) \
  class X##Node;       \
  using X = Ref<X##Node>;
#include <x_macro/stmt.x.h>

class StmtNode : public IRBaseNode {};

}  // namespace domino

#endif  // DOMINO_STMT_H
