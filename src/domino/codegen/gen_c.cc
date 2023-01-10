#include <codegen/gen_c.h>

namespace domino {

namespace codegen {

std::string codegen_c(IRBase tree) {
  CodeGenC gen;
  return gen(tree);
}

}  // namespace codegen

}  // namespace domino