#ifndef DOMINO_ARCH_H
#define DOMINO_ARCH_H

#include <ir_base.h>
#include <ref.h>

namespace domino {

namespace arch {

/// Don't use X_Macro for reference declaration
/// for better debug experience

// #define X_DECL_ARCH(X) \
//   class X##Node;       \
//   using X = Ref<X##Node>;
// #include <x_macro/arch.x.h>

class ArchNode : public IRBaseNode {};

using Arch = Ref<ArchNode>;

enum class CompLevel : int { dL0 = 0, dL1 = 1, dL2 = 2, dL3 = 3, dL4 = 4, dL5 = 5, dUNKNOWN = 255 };

enum class MemoryScope : int {
  dRegister = 0,
  dLocal = 1,
  dScratchpad = 2,
  dShared = 3,
  dGloabl = 4,
  dUNKNOWN = 255
};

}  // namespace arch

}  // namespace domino

#endif  // DOMINO_ARCH_H