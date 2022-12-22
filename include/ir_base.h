#ifndef DOMINO_IR_BASE_H
#define DOMINO_IR_BASE_H

#include <ref.h>

namespace domino {

class IRBaseNode {
 public:
  virtual ~IRBaseNode() = default;
  operator std::string() const {
    return "IRBaseNode()";
  }
};
typedef Ref<IRBaseNode> IRBase;

std::ostream& operator<<(std::ostream&, IRBase);
std::string repr(IRBase);

}  // namespace domino

#endif
