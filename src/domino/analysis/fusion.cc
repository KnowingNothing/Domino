#include <analysis/fusion.h>

#include <unordered_set>

namespace domino {

namespace analysis {

#define MAX(a, b) ((a) > (b) ? (a) : (b))

class TensorVarFinder : public IRVisitor<> {
 protected:
  void ImplVisit(Var op) override {
    if (op == var_) {
      table_[op] = true;
    }
  }

  void ImplVisit(AtomBlock op) override {
    Visit(op->getStmt());
    if (table_[op->getStmt()]) {
      table_[op] = true;
    }
  }

  void ImplVisit(Evaluate op) override {
    Visit(op->expr);
    if (table_[op->expr]) {
      table_[op] = true;
    }
  }

  void ImplVisit(arch::ComputeLevel op) override {
    Visit(op->block);
    if (table_[op->block]) {
      table_[op] = true;
    }
  }

 private:
  std::unordered_map<IRBase, bool> table_;
  Var var_;

 public:
  TensorVarFinder(Var var) : var_(var) {}
  void Visit(IRBase op) override {
    if (table_.count(op)) {
      return;
    } else {
      IRVisitor<>::Visit(op);
    }
  }
  bool Find(IRBase op) {
    Visit(op);
    return table_[op];
  }
};

class MemLevelFinder : public IRVisitor<> {
 protected:
  void ImplVisit(arch::ComputeLevel op) override {
    TensorVarFinder finder(var_);
    table_[op] = finder.Find(op);
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

class CountTilableLevels : public IRVisitor<> {
 protected:
  void ImplVisit(Var op) override {
    if (op == tensor_var_) {
      table_[op] = true;
    }
  }

  void ImplVisit(Iterator op) override { tiled_[op] = true; }

  void ImplVisit(ExprList op) override {
    bool has_tensor = false;
    bool tiled = false;
    for (auto v : op->value_list) {
      Visit(v);
      has_tensor |= table_[v];
      tiled |= tiled_[v];
    }
    table_[op] = has_tensor;
    tiled_[op] = tiled;
  }

  void ImplVisit(AtomBlock op) override {
    Visit(op->getStmt());
    table_[op] = table_[op->getStmt()];
    tiled_[op] = tiled_[op->getStmt()];
  }

  void ImplVisit(SeqBlock op) override {
    Visit(op->first);
    Visit(op->second);
    table_[op] = table_[op->first] || table_[op->second];
    tiled_[op] = tiled_[op->first] || tiled_[op->second];
  }

  void ImplVisit(Evaluate op) override {
    Visit(op->expr);
    table_[op] = table_[op->expr];
    tiled_[op] = tiled_[op->expr];
  }

  void ImplVisit(arch::ComputeLevel op) override {
    Visit(op->block);
    table_[op] = table_[op->block];
  }

  void ImplVisit(arch::MemoryLevel op) override {
    bool has_tensor = false;
    Visit(op->block);
    bool tiled = tiled_[op->block];
    for (auto sub : op->sub_levels) {
      Visit(sub);
      has_tensor |= table_[sub];
    }
    if (has_tensor && !tiled) {
      ans_ += 1;
    }
    table_[op] = has_tensor;
    tiled_[op] = tiled;
  }

 public:
  CountTilableLevels(Var tensor_var) : tensor_var_(tensor_var) {}

  int Get() const { return ans_; }

  void Visit(IRBase op) override {
    if (table_.count(op) && tiled_.count(op)) {
      return;
    }
    table_[op] = false;
    tiled_[op] = false;
    IRVisitor<>::Visit(op);
  }

 private:
  Var tensor_var_;
  std::unordered_map<IRBase, bool> table_;
  std::unordered_map<IRBase, bool> tiled_;
  int ans_{0};
};

class MemoryTiler : public ArchMutator {
 protected:
  arch::Arch ImplVisit(arch::MemoryLevel op) override {
    bool has_tensor = false;
    std::vector<arch::Arch> new_sub_levels;
    for (auto sub : op->sub_levels) {
      arch::Arch new_sub = Visit(sub);
      has_tensor |= table_[sub];
      new_sub_levels.push_back(new_sub);
    }
    Block new_block = VisitBlock(op->block);
    if (has_tensor) {
      if (level_counter_ < tiles_.size()) {
        BlockMutator block_mutator(std::make_shared<MakeTile>(tiles_[level_counter_]));
        new_block = block_mutator(op->block);
      }
      level_counter_ += 1;
    }
    table_[op] = has_tensor;
    return arch::MemoryLevel::make(op->memory_level, new_block, new_sub_levels);
  }

  arch::Arch ImplVisit(arch::ComputeLevel op) override {
    TensorVarFinder finder(tensor_var_);
    table_[op] = finder.Find(op);
    return op;
  }

 public:
  MemoryTiler(Var tensor_var, std::vector<std::vector<Iterator>> tiles)
      : tensor_var_(tensor_var), tiles_(tiles) {}

  arch::Arch Visit(IRBase op) override {
    if (!table_.count(op)) {
      table_[op] = false;
    }
    return ArchMutator::Visit(op);
  }

 private:
  Var tensor_var_;
  std::vector<std::vector<Iterator>> tiles_;
  std::unordered_map<IRBase, bool> table_;
  int level_counter_ = 0;

  class MakeTile : public StmtMutator {
   protected:
    Stmt ImplVisit(Evaluate op) override {
      Expr expr = VisitExpr(op->expr);
      ConstInt as_int = expr.as<ConstIntNode>();
      if (as_int.defined()) {
        // replace null (Evaluate(0)) with Evaluate(ExprList[it1, it2, ...])
        std::vector<Expr> tmp;
        for (auto it : this->iters_) {
          tmp.push_back(it);
        }
        return Evaluate::make(ExprList::make(tmp));
      } else {
        return Evaluate::make(expr);
      }
    }

   public:
    MakeTile(std::vector<Iterator> iters) : iters_(iters) {}

   private:
    std::vector<Iterator> iters_;
  };
};

class LeastCommonMemoryLevelFinder : public IRVisitor<> {
 protected:
  void ImplVisit(arch::MemoryLevel op) override {
    for (auto sub : op->sub_levels) {
      Visit(sub);
      for (int i = 0; i < tensor_vars_.size(); ++i) {
        has_tensor_table_[i][op] |= has_tensor_table_[i][sub];
      }
    }
    for (int i = 0; i < tensor_vars_.size(); ++i) {
      if (!has_tensor_table_[i][op]) {
        return;
      }
    }
    if (!ans_.defined()) {
      ans_ = op;
    }
  }

  void ImplVisit(arch::ComputeLevel op) override {
    for (int i = 0; i < tensor_vars_.size(); ++i) {
      Var v = tensor_vars_[i];
      TensorVarFinder finder(v);
      has_tensor_table_[i][op] = finder.Find(op);
    }
  }

 public:
  void Visit(IRBase op) override {
    for (auto& m : has_tensor_table_) {
      m[op] = false;
    }
    IRVisitor<>::Visit(op);
  }

  LeastCommonMemoryLevelFinder(std::vector<Var> tensor_vars)
      : tensor_vars_(std::move(tensor_vars)) {
    for (auto v : tensor_vars_) {
      has_tensor_table_.push_back(std::unordered_map<IRBase, bool>());
    }
  }

  arch::MemoryLevel Find(IRBase op) {
    Visit(op);
    return ans_;
  }

 private:
  std::vector<Var> tensor_vars_;
  std::vector<std::unordered_map<IRBase, bool>> has_tensor_table_;
  arch::MemoryLevel ans_ = nullptr;
};

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
  new_tree->var_map = this->var_map;
  new_tree->bounds = this->bounds;
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

  new_tree->var_map = this->var_map;
  for (auto kv : other->var_map) {
    new_tree->var_map[kv.first] = kv.second;
  }
  new_tree->bounds = this->bounds;
  for (auto kv : other->bounds) {
    new_tree->bounds[kv.first] = kv.second;
  }
  return new_tree;
}

int MemoryLevelTreeNode::GetAvailableLevels(Var tensor_var) const {
  CountTilableLevels counter(tensor_var);
  counter(this->root);
  return counter.Get();
}

MemoryLevelTree MemoryLevelTreeNode::MemoryTiling(Var tensor_var, std::vector<Var> loop_vars,
                                                  std::vector<std::vector<Iterator>> tiles) const {
  ArchMutator mutator;
  arch::Arch new_root = mutator(this->root);

  MemoryTiler tiler(tensor_var, tiles);
  new_root = tiler(new_root);
  MemoryLevelTree new_tree =
      MemoryLevelTree::make(new_root, this->merged, this->initial_levels, this->tensor_var);
  // inherit var map for tiling
  new_tree->var_map = this->var_map;
  int num_loops = loop_vars.size();
  std::vector<std::vector<Var>> var_list(num_loops, std::vector<Var>());
  for (auto tile : tiles) {
    ASSERT((int)tile.size() == num_loops);
    for (int i = 0; i < num_loops; ++i) {
      var_list[i].push_back(tile[i]->var);
    }
  }
  for (int i = 0; i < num_loops; ++i) {
    new_tree->var_map[loop_vars[i]] =
        ExprList::make(std::vector<Expr>(var_list[i].begin(), var_list[i].end()));
  }
  // inherit bounds
  new_tree->bounds = this->bounds;
  for (auto tile : tiles) {
    for (int i = 0; i < num_loops; ++i) {
      Var loop_var = loop_vars[i];
      ASSERT(this->bounds.count(loop_var));
      Expr start = this->bounds.at(loop_var)->beg;
      /// The tiles should be normalized
      ConstInt as_int = tile[i]->range->beg.as<ConstIntNode>();
      ASSERT(as_int.defined() && as_int->value == 0);
      as_int = tile[i]->range->step.as<ConstIntNode>();
      ASSERT(as_int.defined() && as_int->value == 1);
      /// This is the real bound for the sub-loop
      Range new_bound = Range::make(start, tile[i]->range->extent, tile[i]->range->step);
      new_tree->bounds[tile[i]->var] = tile[i]->range;
    }
  }
  return new_tree;
}

arch::MemoryLevel MemoryLevelTreeNode::LeastCommonAncestor(Var tensor_var1, Var tensor_var2) const {
  LeastCommonMemoryLevelFinder finder({tensor_var1, tensor_var2});
  arch::MemoryLevel ret = finder.Find(this->root);
  if (!ret.defined()) {
    throw std::runtime_error(fmt::format("Can't find the common ancestor for {} and {}",
                                         std::string(tensor_var1), std::string(tensor_var2)));
  }
  return ret;
}

void MemoryLevelTreeNode::SetBounds(std::unordered_map<Var, Range> new_bounds) {
  for (auto [k, v] : new_bounds) {
    // if (this->bounds.count(k)) {
    //   WARN << fmt::format(
    //       "The bound of {} has already been set to {}, but now want to set it to {}",
    //       std::string(k), std::string(this->bounds.at(k)), std::string(v));
    // }
    this->bounds[k] = v;
  }
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
    std::unordered_map<Var, MergePoint> merge_points, std::vector<MemoryLevelTree>& ans,
    std::unordered_set<std::string>& cache) {
  if (cur_tensor_var_id == tensor_vars.size()) {
    std::string str = repr(cur_tree->root);
    if (cache.count(str)) {
      return;
    }
    cache.insert(str);
    ans.push_back(cur_tree);
    return;
  }
  Var cur_tensor_var = tensor_vars[cur_tensor_var_id];
  if (!compute_tensor_mask[cur_tensor_var_id]) {
    recursiveGenerateMergedMemoryLevelTrees(cur_tensor_var_id + 1, levels, initial_trees,
                                            tensor_vars, compute_tensor_mask, tensor_var_dominators,
                                            cur_tree, merge_points, ans, cache);
  } else {
    if (tensor_var_dominators.count(cur_tensor_var)) {
      Var upper_dom = cur_tensor_var;
      int lower_level_id = 0;
      Var dom = tensor_var_dominators.at(cur_tensor_var);
      while (dom != upper_dom) {
        upper_dom = dom;
        if (merge_points.count(dom)) {
          MergePoint point = merge_points[dom];
          for (int l = lower_level_id; l < MAX(1, point.level_id); ++l) {
            int level = levels[l];
            MemoryLevelTree new_tree =
                cur_tree->Merge(initial_trees[cur_tensor_var_id], dom, level);
            merge_points[cur_tensor_var] = MergePoint(dom, l);
            recursiveGenerateMergedMemoryLevelTrees(
                cur_tensor_var_id + 1, levels, initial_trees, tensor_vars, compute_tensor_mask,
                tensor_var_dominators, new_tree, merge_points, ans, cache);
            merge_points.erase(cur_tensor_var);
          }
          dom = point.tensor_var;
          lower_level_id = MAX(lower_level_id, point.level_id);
        }
      }
    } else {
      throw std::runtime_error("Don't know how to handle undominated cases.");
    }
  }
}

std::vector<MemoryLevelTree> generateMergedMemoryLevelTrees(
    std::vector<Var> tensor_vars, std::vector<std::unordered_map<Var, Range>> initial_bounds,
    std::vector<bool> compute_tensor_mask, std::unordered_map<Var, Var> tensor_var_dominators,
    std::vector<int> levels) {
  ASSERT(tensor_vars.size() > 0);
  std::vector<MemoryLevelTree> init_trees;
  ASSERT(tensor_vars.size() == initial_bounds.size());
  for (int i = 0; i < tensor_vars.size(); ++i) {
    Var t = tensor_vars[i];
    auto bounds = initial_bounds[i];
    init_trees.push_back(MemoryLevelTree::make(levels, t, bounds));
  }
  std::unordered_map<Var, MergePoint> merge_points;
  merge_points[tensor_vars[0]] = MergePoint(tensor_vars[0], levels.size());
  std::vector<MemoryLevelTree> ret;
  std::unordered_set<std::string> cache;
  recursiveGenerateMergedMemoryLevelTrees(1, levels, init_trees, tensor_vars, compute_tensor_mask,
                                          tensor_var_dominators, init_trees[0], merge_points, ret,
                                          cache);
  return ret;
}

}  // namespace analysis

}  // namespace domino