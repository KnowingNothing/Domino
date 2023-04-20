#include <codegen/gen_tileflow.h>
#include <visitor.h>

#include <sstream>
#include <unordered_set>

namespace domino {
using namespace domino::arch;
namespace codegen {

namespace {
class GatherVars : public IRVisitor<> {
 protected:
  using IRVisitor<>::ImplVisit;
  void ImplVisit(Var op) override { var_table_.insert(op); }

 private:
  std::unordered_set<Var> var_table_;

 public:
  using IRVisitor<>::Visit;
  std::unordered_set<Var> all_vars() { return var_table_; }
};
}  // namespace

std::string CodeGenTileFlow::ImplVisit(ComputeLevel op) {
  ASSERT(parent_pointers_.count(op));
  Arch parent = parent_pointers_.at(op);
  ASSERT(loop_table_.count(parent));
  std::vector<Iterator> loops = loop_table_.at(parent);
  std::ostringstream oss;
  std::string indent = make_indent();
  std::string name = "Produce";
  if (op->produce_var.defined()) {
    name += Visit(op->produce_var);
  }
  GatherVars gather;
  gather.Visit(op->block);
  std::unordered_set<Var> var_set = gather.all_vars();
  oss << indent << "- node-type: Op\n"
      << indent << "  name: " << name << "\n"
      << indent << "  binding:";
  for (int i = 0, n = loops.size(); i < n; ++i) {
    if (var_set.count(loops[i]->var))
      oss << " " << renaming(Visit(loops[i]->var)) << ":" << renaming(Visit(loops[i]->var));
  }
  oss << "\n";
  return oss.str();
}

std::string CodeGenTileFlow::ImplVisit(MemoryLevel op) {
  bool is_top = top_;
  top_ = false;
  std::vector<Iterator> loops;
  if (parent_pointers_.count(op)) {
    Arch parent = parent_pointers_.at(op);
    ASSERT(loop_table_.count(parent));
    loops = loop_table_.at(parent);
  }
  std::unordered_set<Var> visit;
  for (auto iter : loops) {
    visit.insert(iter->var);
  }
  std::ostringstream oss;
  std::string indent = make_indent();
  std::vector<std::string> sub_level_strs;

  AtomBlock block = op->block.as<AtomBlockNode>();
  if (block.defined()) {
    Evaluate eval = block->getStmt().as<EvaluateNode>();
    if (eval.defined()) {
      ExprList list = eval->expr.as<ExprListNode>();
      ConstInt zero = eval->expr.as<ConstIntNode>();
      if (list.defined()) {
        std::ostringstream factors;
        std::ostringstream perm;
        std::unordered_map<std::string, int> factor_map;
        for (auto it : list->value_list) {
          Iterator iter = it.as<IteratorNode>();
          ASSERT(iter.defined());
          std::string name = renaming(Visit(iter->var));
          ConstInt factor = iter->range->extent.as<ConstIntNode>();
          if (factor.defined()) {
            if (factor_map.count(name)) {
              factor_map[name] *= factor->value;
            } else {
              factor_map[name] = factor->value;
            }
          } else {
            factor_map[name] = -1;
          }

          if (!visit.count(iter->var)) {
            loops.push_back(iter);
            visit.insert(iter->var);
          }
        }
        for (auto it : list->value_list) {
          Iterator iter = it.as<IteratorNode>();
          ASSERT(iter.defined());
          std::string name = renaming(Visit(iter->var));
          if (factor_map.count(name)) {
            factors << " " << name << "=";
            if (factor_map[name] == -1) {
              factors << "?";
            } else {
              factors << factor_map[name];
            }
            perm << renaming(Visit(iter->var));
            factor_map.erase(name);
          }
        }
        loop_table_[op] = loops;
        increase_indent();
        if (!is_top) increase_indent();
        for (auto sub : op->sub_levels) {
          parent_pointers_[sub] = op;
          sub_level_strs.push_back(Visit(sub));
        }
        if (!is_top) decrease_indent();
        decrease_indent();
        if (is_top) {
          oss << indent << "node-type: Tile\n"
              << indent << "type: " << op->annotation << "\n"
              << indent << "factors:" << factors.str() << "\n"
              << indent << "permutation: " << perm.str() << "\n"
              << indent << "target: L" << Visit(op->memory_level) << "\n";
        } else {
          oss << indent << "- node-type: Tile\n"
              << indent << "  type: " << op->annotation << "\n"
              << indent << "  factors:" << factors.str() << "\n"
              << indent << "  permutation: " << perm.str() << "\n"
              << indent << "  target: L" << Visit(op->memory_level) << "\n";
        }

        if (op->sub_levels.size()) {
          if (is_top)
            oss << "\n" << indent << "subtree:\n";
          else
            oss << "\n" << indent << "  subtree:\n";
          for (auto sub : sub_level_strs) {
            oss << sub;
          }
        }
      } else {
        ASSERT(zero.defined() && zero->value == 0);
        loop_table_[op] = loops;
        increase_indent();
        if (!is_top) increase_indent();
        for (auto sub : op->sub_levels) {
          parent_pointers_[sub] = op;
          sub_level_strs.push_back(Visit(sub));
        }
        if (!is_top) decrease_indent();
        decrease_indent();
        if (is_top) {
          oss << indent << "node-type: Scope\n" << indent << "  type: " << op->scope << "\n";
        } else {
          oss << indent << "- node-type: Scope\n" << indent << "  type: " << op->scope << "\n";
        }
        if (op->sub_levels.size()) {
          if (is_top)
            oss << "\n" << indent << "subtree:\n";
          else
            oss << "\n" << indent << "  subtree:\n";
          for (auto sub : sub_level_strs) {
            oss << sub;
          }
        }
      }
    }
  }
  return oss.str();
}

std::string codegen_tileflow(IRBase tree) {
  CodeGenTileFlow gen;
  return "mapping:\n" + gen(tree);
}
}  // namespace codegen
}  // namespace domino