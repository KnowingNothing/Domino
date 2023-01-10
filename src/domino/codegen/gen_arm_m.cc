#include <codegen/gen_arm_m.h>

namespace domino {

namespace codegen {

std::string codegen_arm_m(IRBase tree) {
  CodeGenARM_M gen;
  return gen(tree);
}

}  // namespace codegen

}  // namespace domino