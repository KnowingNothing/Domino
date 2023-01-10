#ifndef DOMINO_CODEGEN_GEN_C_H
#define DOMINO_CODEGEN_GEN_C_H

#include <codegen/gen_base.h>

namespace domino {

namespace codegen {

class CodeGenC : public CodeGenBase {};

std::string codegen_c(IRBase tree);

}  // namespace codegen

}  // namespace domino

#endif  // DOMINO_CODEGEN_GEN_C_H