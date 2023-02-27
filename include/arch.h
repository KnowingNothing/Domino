#ifndef DOMINO_ARCH_H
#define DOMINO_ARCH_H

#include <block.h>
#include <expr.h>
#include <ir_base.h>
#include <ref.h>

#include <vector>

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

class MemoryLevelNode : public ArchNode {
 public:
  MemoryLevelNode(ConstInt level, Block b, std::vector<Arch> sub)
      : memory_level(std::move(level)), block(std::move(b)), sub_levels(std::move(sub)) {
    ASSERT(this->memory_level.defined());
    ASSERT(this->block.defined());
  }

  MemoryLevelNode(int level, Block b, std::vector<Arch> sub)
      : memory_level(ConstInt::make(level)), block(std::move(b)), sub_levels(std::move(sub)) {
    ASSERT(this->memory_level.defined());
    ASSERT(this->block.defined());
  }

  ConstInt memory_level;
  Block block;
  std::vector<Arch> sub_levels;
};

using MemoryLevel = Ref<MemoryLevelNode>;

class ComputeLevelNode : public ArchNode {
 public:
  ComputeLevelNode(ConstInt level, Block b, std::vector<Arch> sub)
      : compute_level(std::move(level)), block(std::move(b)), sub_levels(std::move(sub)) {
    ASSERT(this->compute_level.defined());
    ASSERT(this->block.defined());
  }

  ComputeLevelNode(int level, Block b, std::vector<Arch> sub)
      : compute_level(ConstInt::make(level)), block(std::move(b)), sub_levels(std::move(sub)) {
    ASSERT(this->compute_level.defined());
    ASSERT(this->block.defined());
  }

  ConstInt compute_level;
  Block block;
  std::vector<Arch> sub_levels;
};

using ComputeLevel = Ref<ComputeLevelNode>;

}  // namespace arch

}  // namespace domino

#endif  // DOMINO_ARCH_H