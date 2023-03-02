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

class MemLevelFinder : public IRVisitor<> {
 protected:
  void ImplVisit(arch::ComputeLevel op) override {
    AtomBlock as_atom = op->block.as<AtomBlockNode>();
    if (as_atom.defined()) {
      Evaluate as_eval = as_atom->getStmt().as<EvaluateNode>();
      if (as_eval.defined()) {
        Var as_var = as_eval->expr.as<VarNode>();
        if (as_var.defined() && as_var == var_) {
          table_[op] = true;
        }
      }
    }
  }

  void ImplVisit(arch::MemoryLevel op) override {
    bool has_tensor = false;
    for (auto sub : op->sub_levels) {
      Visit(sub);
      if (table_.count(sub)) {
        has_tensor |= table_.at(sub);
      }
    }
    if (has_tensor && op->memory_level->value == level_) {
      ans_ = op;
    }
    table_[op] = has_tensor;
  }

 private:
  std::unordered_map<IRBase, bool> table_;
  int level_;
  Var var_;
  arch::MemoryLevel ans_;

 public:
  MemLevelFinder(int level, Var var) : level_(level), var_(var) {}
  void Visit(IRBase op) override {
    if (table_.count(op)) {
      return;
    } else {
      IRVisitor<>::Visit(op);
    }
  }
  arch::MemoryLevel Get() { return ans_; }
};

class MemoryLevelTreeNode {
 public:
  MemoryLevelTreeNode(std::vector<int> levels, Var tensor_var) {
    ASSERT(levels.size() > 0) << "Architecture levels should be at least 1.";
    std::vector<arch::Arch> tmp;
    arch::Arch cur_level = arch::ComputeLevel::make(levels[0], MAKE_BLOCK(tensor_var), tmp);
    for (auto l : levels) {
      std::vector<arch::Arch> sub_levels = {cur_level};
      cur_level = arch::MemoryLevel::make(l, MAKE_NULL, sub_levels);
    }
    this->root = cur_level;
  }
  MemoryLevelTreeNode(arch::Arch root, bool merged, std::vector<int> initial_levels, Var tensor_var)
      : root(std::move(root)),
        merged(merged),
        initial_levels(std::move(initial_levels)),
        tensor_var(std::move(tensor_var)) {}

  MemoryLevelTree Cut(int level) const;

  MemoryLevelTree Merge(MemoryLevelTree other, Var tensor_var, int level) const;

  arch::Arch root;
  bool merged{false};
  std::vector<int> initial_levels;
  Var tensor_var;
};

/**
 * Generate all the possible fusion trees
 *
 * \param tensor_vars: toposorted tensor vars from consumers to producers
 * \param compute_tensor_mask: mask if the tensor is result of computation
 * \param tensor_var_dominators: the immediate dominators of each tensor var
 * \param levels: the level from 0 to N of the memory
 */
std::vector<MemoryLevelTree> generateMergedMemoryLevelTrees(
    std::vector<Var> tensor_vars, std::vector<bool> compute_tensor_mask,
    std::unordered_map<Var, Var> tensor_var_dominators, std::vector<int> levels);

}  // namespace analysis

}  // namespace domino

#endif  // DOMINO_ANALYSIS_FUSION_H