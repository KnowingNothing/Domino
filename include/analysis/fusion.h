#ifndef DOMINO_ANALYSIS_FUSION_H
#define DOMINO_ANALYSIS_FUSION_H

#include <arch.h>
#include <block.h>
#include <expr.h>
#include <mutator.h>
#include <ref.h>
#include <stmt.h>
#include <visitor.h>

#include <unordered_map>
#include <vector>

namespace domino {

namespace analysis {

#define MAKE_NULL AtomBlock::make(Evaluate::make(ConstInt::make(0)))
#define MAKE_BLOCK(x) AtomBlock::make(Evaluate::make(x))

class MemoryLevelTreeNode;
using MemoryLevelTree = Ref<MemoryLevelTreeNode>;

class MemoryLevelTreeNode {
 public:
  MemoryLevelTreeNode(std::vector<int> levels, Var tensor_var,
                      std::unordered_map<Var, Range> initial_bounds) {
    ASSERT(levels.size() > 0) << "Architecture levels should be at least 1.";
    std::vector<arch::Arch> tmp;
    arch::Arch cur_level = arch::ComputeLevel::make(levels[0], MAKE_BLOCK(tensor_var), tmp);
    for (auto l : levels) {
      std::vector<arch::Arch> sub_levels = {cur_level};
      cur_level = arch::MemoryLevel::make(l, MAKE_NULL, sub_levels);
    }
    this->root = cur_level;
    this->bounds = initial_bounds;
  }
  MemoryLevelTreeNode(arch::Arch root, bool merged, std::vector<int> initial_levels, Var tensor_var)
      : root(std::move(root)),
        merged(merged),
        initial_levels(std::move(initial_levels)),
        tensor_var(std::move(tensor_var)) {}

  /**
   * Cut the current tree from a certain level.
   * Only consider simple linear memory level tree that has not been merged.
   */
  MemoryLevelTree Cut(int level) const;

  /**
   * Merge a simple linear memory level tree to another complex merged tree.
   * \param other: the memory level tree to be merged into current tree
   * \param tensor_var: used to find the path to merge into
   * \param level: the merge level
   */
  MemoryLevelTree Merge(MemoryLevelTree other, Var tensor_var, int level) const;

  /**
   * Tell how many levels of memory are needed for tiling for a given tensor_var
   */
  int GetAvailableLevels(Var tensor_var) const;

  /**
   * Perform tiling for memory levels
   * \param tensor_var: the tiling target tensor
   * \param loop_vars: the loop vars to tile
   * \param tiles: the tiling factors in format of Iterators
   */
  MemoryLevelTree MemoryTiling(Var tensor_var, std::vector<Var> loop_vars,
                               std::vector<std::vector<Iterator>> tiles) const;

  /**
   * Find the least common ancestor for two tensor vars
   */
  arch::MemoryLevel LeastCommonAncestor(Var tensor_var1, Var tensor_var2) const;

  /**
   * Set the bounds.
   */
  void SetBounds(std::unordered_map<Var, Range> new_bounds);

  arch::Arch root;
  bool merged{false};
  std::vector<int> initial_levels;
  Var tensor_var;
  std::unordered_map<Var, Expr> var_map;
  std::unordered_map<Var, Range> bounds;
};  // namespace analysis

/**
 * Generate all the possible fusion trees
 *
 * \param tensor_vars: toposorted tensor vars from consumers to producers
 * \param initial_bounds: the initial bounds
 * \param compute_tensor_mask: mask if the tensor is result of computation
 * \param tensor_var_dominators: the immediate dominators of each tensor var
 * \param levels: the level from 0 to N of the memory
 */
std::vector<MemoryLevelTree> generateMergedMemoryLevelTrees(
    std::vector<Var> tensor_vars, std::vector<std::unordered_map<Var, Range>> initial_bounds,
    std::vector<bool> compute_tensor_mask, std::unordered_map<Var, Var> tensor_var_dominators,
    std::vector<int> levels);

}  // namespace analysis

}  // namespace domino

#endif  // DOMINO_ANALYSIS_FUSION_H