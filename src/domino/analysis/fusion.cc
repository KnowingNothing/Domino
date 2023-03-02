#include <analysis/fusion.h>

namespace domino {

namespace analysis {

#define MAX(a, b) ((a) > (b) ? (a) : (b))

MemoryLevelTree MemoryLevelTreeNode::Cut(int level) const {
  ASSERT(this->root.defined()) << "Can't operate on unconstructed trees.";
  ArchMutator mutator;
  arch::Arch new_root = mutator(this->root);
  while (new_root.as<arch::MemoryLevelNode>().defined()) {
    MemoryLevel mem = new_root.as<arch::MemoryLevelNode>();
    ASSERT(mem->sub_levels.size() > 0);
    new_root = mem->sub_levels[0];
    if (mem->memory_level->value == level) {
      break;
    }
  }
  MemoryLevelTree new_tree =
      MemoryLevelTree::make(new_root, this->merged, this->initial_levels, this->tensor_var);
  return new_tree;
}

MemoryLevelTree MemoryLevelTreeNode::Merge(MemoryLevelTree other, Var tensor_var, int level) const {
  ASSERT(!other->merged) << "Can't merge a complex tree to another tree.";
  ArchMutator mutator;
  arch::Arch new_root = mutator(this->root);

  MemLevelFinder finder(level, tensor_var);
  finder(new_root);
  arch::MemoryLevel position = finder.Get();
  ASSERT(position.defined()) << "Can't find the position to merge.";
  other = other->Cut(level);
  std::vector<arch::Arch> new_subs = {other->root};
  for (auto s : position->sub_levels) {
    new_subs.push_back(s);
  }
  position->sub_levels = new_subs;
  MemoryLevelTree new_tree =
      MemoryLevelTree::make(new_root, this->merged, this->initial_levels, this->tensor_var);
  return new_tree;
}

class MergePoint {
 public:
  MergePoint() {}
  MergePoint(const MergePoint&) = default;
  MergePoint(Var tensor_var, int level_id)
      : tensor_var(std::move(tensor_var)), level_id(level_id) {}
  Var tensor_var;
  int level_id;
};

void recursiveGenerateMergedMemoryLevelTrees(
    int cur_tensor_var_id, const std::vector<int>& levels,
    const std::vector<MemoryLevelTree>& initial_trees, const std::vector<Var>& tensor_vars,
    const std::vector<bool>& compute_tensor_mask,
    const std::unordered_map<Var, Var>& tensor_var_dominators, MemoryLevelTree cur_tree,
    std::unordered_map<Var, MergePoint> merge_points, std::vector<MemoryLevelTree>& ans) {
  if (cur_tensor_var_id == tensor_vars.size()) {
    ans.push_back(cur_tree);
    return;
  }
  Var cur_tensor_var = tensor_vars[cur_tensor_var_id];
  if (!compute_tensor_mask[cur_tensor_var_id]) {
    recursiveGenerateMergedMemoryLevelTrees(cur_tensor_var_id + 1, levels, initial_trees,
                                            tensor_vars, compute_tensor_mask, tensor_var_dominators,
                                            cur_tree, merge_points, ans);
  } else {
    if (tensor_var_dominators.count(cur_tensor_var)) {
      Var upper_dom = cur_tensor_var;
      Var dom = tensor_var_dominators.at(cur_tensor_var);
      while (dom != upper_dom) {
        upper_dom = dom;
        if (merge_points.count(dom)) {
          MergePoint point = merge_points[dom];
          for (int l = 0; l < MAX(1, point.level_id); ++l) {
            int level = levels[l];
            MemoryLevelTree new_tree =
                cur_tree->Merge(initial_trees[cur_tensor_var_id], dom, level);
            merge_points[cur_tensor_var] = MergePoint(dom, l);
            recursiveGenerateMergedMemoryLevelTrees(
                cur_tensor_var_id + 1, levels, initial_trees, tensor_vars, compute_tensor_mask,
                tensor_var_dominators, new_tree, merge_points, ans);
            merge_points.erase(cur_tensor_var);
          }
          dom = point.tensor_var;
        }
      }
    } else {
      throw std::runtime_error("Don't know how to handle undominated cases.");
    }
  }
}

std::vector<MemoryLevelTree> generateMergedMemoryLevelTrees(
    std::vector<Var> tensor_vars, std::vector<bool> compute_tensor_mask,
    std::unordered_map<Var, Var> tensor_var_dominators, std::vector<int> levels) {
  ASSERT(tensor_vars.size() > 0);
  std::vector<MemoryLevelTree> init_trees;
  for (auto t : tensor_vars) {
    init_trees.push_back(MemoryLevelTree::make(levels, t));
  }
  std::unordered_map<Var, MergePoint> merge_points;
  merge_points[tensor_vars[0]] = MergePoint(tensor_vars[0], levels.size() - 1);
  std::vector<MemoryLevelTree> ret;
  recursiveGenerateMergedMemoryLevelTrees(1, levels, init_trees, tensor_vars, compute_tensor_mask,
                                          tensor_var_dominators, init_trees[0], merge_points, ret);
  return ret;
}

}  // namespace analysis

}  // namespace domino