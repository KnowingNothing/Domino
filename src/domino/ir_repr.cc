#include <fmt/core.h>
#include <fmt/ranges.h>
#include <ir_base.h>
#include <ir_functor.h>

#include <range/v3/view.hpp>
#include <vector>

namespace domino {

class IRPrinter : public IRFunctor<std::string()> {
 protected:
  /// expressions
  std::string ImplVisit(MemRef mem_ref) override {
    return fmt::format("({}+{})", Visit(mem_ref->var), Visit(mem_ref->offset));
  }

  std::string visit_bin_op(BinExpr bin, std::string op) {
    return fmt::format("({} {} {})", Visit(bin->a), op, Visit(bin->b));
  }

  std::string ImplVisit(Add op) override { return this->visit_bin_op(op, "+"); }

  std::string ImplVisit(Sub op) override { return this->visit_bin_op(op, "-"); }

  std::string ImplVisit(Mul op) override { return this->visit_bin_op(op, "*"); }

  std::string ImplVisit(Div op) override { return this->visit_bin_op(op, "/"); }

  std::string ImplVisit(Mod op) override { return this->visit_bin_op(op, "%"); }

  std::string ImplVisit(FloorDiv op) override { return this->visit_bin_op(op, "//"); }

  std::string ImplVisit(FloorMod op) override {
    return fmt::format("floormod({}, {})", Visit(op->a), Visit(op->b));
  }

  std::string ImplVisit(And op) override { return this->visit_bin_op(op, "&&"); }

  std::string ImplVisit(Or op) override { return this->visit_bin_op(op, "||"); }

  std::string ImplVisit(XOr op) override { return this->visit_bin_op(op, "xor"); }

  std::string ImplVisit(BitAnd op) override { return this->visit_bin_op(op, "&"); }

  std::string ImplVisit(BitOr op) override { return this->visit_bin_op(op, "|"); }

  std::string ImplVisit(BitXOr op) override { return this->visit_bin_op(op, "^"); }

  std::string ImplVisit(GT op) override { return this->visit_bin_op(op, ">"); }

  std::string ImplVisit(GE op) override { return this->visit_bin_op(op, ">="); }

  std::string ImplVisit(LT op) override { return this->visit_bin_op(op, "<"); }

  std::string ImplVisit(LE op) override { return this->visit_bin_op(op, "<="); }

  std::string ImplVisit(EQ op) override { return this->visit_bin_op(op, "=="); }

  std::string ImplVisit(NE op) override { return this->visit_bin_op(op, "!="); }

  std::string ImplVisit(Cast op) override {
    return fmt::format("cast({}, {})", std::string(op->dtype), Visit(op->a));
  }

  std::string ImplVisit(Broadcast op) override {
    return fmt::format("broadcast({}, {})", std::string(op->dtype), Visit(op->a));
  }

  std::string ImplVisit(Neg op) override { return fmt::format("(-{})", Visit(op->a)); }

  std::string ImplVisit(Not op) override { return fmt::format("(!{})", Visit(op->a)); }

  std::string ImplVisit(BitNot op) override { return fmt::format("(~{})", Visit(op->a)); }

  std::string ImplVisit(Ceil op) override { return fmt::format("ceil({})", Visit(op->a)); }

  std::string ImplVisit(Floor op) override { return fmt::format("floor({})", Visit(op->a)); }

  std::string ImplVisit(Select op) override {
    return fmt::format("({} ? {} : {})", Visit(op->a), Visit(op->b), Visit(op->c));
  }

  std::string ImplVisit(Range op) override {
    return fmt::format("range(beg={}, ext={}, step={})", Visit(op->beg), Visit(op->extent),
                       Visit(op->step));
  }

  std::string ImplVisit(ExprList op) override {
    std::vector<std::string> operands;
    for (auto v : op->value_list) {
      operands.push_back(Visit(v));
    }
    return fmt::format("{}", fmt::join(operands, ", "));
  }

  std::string ImplVisit(CondAll op) override { return fmt::format("all({})", Visit(op->phases)); }

  std::string ImplVisit(CondAny op) override { return fmt::format("any({})", Visit(op->phases)); }

  std::string ImplVisit(ConstInt op) override { return fmt::format("{}", op->value); }

  std::string ImplVisit(ConstUInt op) override { return fmt::format("{}", op->value); }

  std::string ImplVisit(ConstFloat op) override { return fmt::format("{}", op->value); }

  std::string ImplVisit(ConstBFloat op) override { return fmt::format("bfloat({})", op->value); }

  std::string ImplVisit(ConstTFloat op) override { return fmt::format("tfloat({})", op->value); }

  std::string ImplVisit(ConstString op) override { return fmt::format("{}", op->value); }

  std::string ImplVisit(Var op) override { return fmt::format("{}", Visit(op->id)); }

  std::string ImplVisit(Iterator op) override {
    return fmt::format("iter_var({}, {}, {})", Visit(op->var), Visit(op->range),
                       iter_type_to_string(op->iter_type));
  }

  std::string ImplVisit(NdLoad op) override {
    return fmt::format("load_n({}, {})", Visit(op->mem_ref), Visit(op->indices));
  }

  std::string ImplVisit(Load op) override {
    return fmt::format("({}[{}])", Visit(op->mem_ref), Visit(op->addr));
  }

  std::string ImplVisit(MapVar op) override {
    return fmt::format("map({}={})", Visit(op->var), Visit(op->expr));
  }

  std::string ImplVisit(Slice op) override {
    std::vector<std::string> strs;
    for (auto v : op->indices) {
      strs.push_back(fmt::format("{}:{}:{}", Visit(v->beg), Visit(v->extent), Visit(v->step)));
    }
    return fmt::format("{}", fmt::join(strs, ", "));
  }

  std::string ImplVisit(MemSlice op) override {
    return fmt::format("({}+{})[{}]", Visit(op->var), Visit(op->offset), Visit(op->slice));
  }

  std::string ImplVisit(Call op) override {
    return fmt::format("{}({})", Visit(op->func), Visit(op->args));
  }

  /// statements
  std::string ImplVisit(NdStore op) override {
    return fmt::format("store_n({}, {}, {});", Visit(op->mem_ref), Visit(op->indices),
                       Visit(op->values));
  }

  std::string ImplVisit(Store op) override {
    return fmt::format("{}[{}] = {};", Visit(op->mem_ref), Visit(op->addr), Visit(op->value));
  }

  std::string ImplVisit(Evaluate op) override { return fmt::format("({});", Visit(op->expr)); }

  /// blocks
  std::string ImplVisit(AttrBlock op) override {
    std::string body_str = Visit(op->body);
    std::string ind = make_indent();
    return fmt::format("{}attr({}, {}={})\n{}", ind, Visit(op->obj), Visit(op->key),
                       Visit(op->value), body_str);
  }

  std::string ImplVisit(NdForBlock op) override {
    /// increase indent
    increase_indent();
    std::string body_str = Visit(op->body);
    /// decrease indent
    decrease_indent();
    std::vector<std::string> inits;
    std::vector<std::string> conds;
    std::vector<std::string> steps;
    std::string ind = make_indent();
    for (auto v : op->iters) {
      inits.push_back(fmt::format("{}={}", Visit(v->var), Visit(v->range->beg)));
      conds.push_back(fmt::format("{}<{}", Visit(v->var), Visit(v->range->extent)));
      steps.push_back(fmt::format("{}+={}", Visit(v->var), Visit(v->range->step)));
    }
    std::string left = "{";
    std::string right = "}";
    return fmt::format("{}{}_for_n({}; {}; {}) {}\n{}{}{}\n", ind, Visit(op->compute_level),
                       fmt::join(inits, ", "), fmt::join(conds, ", "), fmt::join(steps, ", "), left,
                       body_str, ind, right);
  }

  std::string ImplVisit(ForBlock op) override {
    /// increase indent
    increase_indent();
    std::string body_str = Visit(op->body);
    /// decrease indent
    decrease_indent();
    std::string init = fmt::format("{}={}", Visit(op->iter->var), Visit(op->iter->range->beg));
    std::string cond = fmt::format("{}<{}", Visit(op->iter->var), Visit(op->iter->range->extent));
    std::string step = fmt::format("{}+={}", Visit(op->iter->var), Visit(op->iter->range->step));
    std::string ind = make_indent();

    std::string left = "{";
    std::string right = "}";
    return fmt::format("{}{}_for({}; {}; {}) {}\n{}{}{}\n", ind, Visit(op->compute_level), init,
                       cond, step, left, body_str, ind, right);
  }

  std::string ImplVisit(BranchBlock op) override {
    std::string cond = Visit(op->cond);
    /// increase indent
    increase_indent();
    std::string true_branch = Visit(op->true_branch);
    std::string false_branch = Visit(op->false_branch);
    /// decrease indent
    decrease_indent();
    std::string ind = make_indent();

    std::string left = "{";
    std::string right = "}";
    return fmt::format("{}if({}) {}\n{}{}{} else {}\n{}{}{}\n", ind, cond, left, true_branch, ind,
                       right, left, false_branch, ind, right);
  }

  std::string ImplVisit(SeqBlock op) override {
    std::string first_str = Visit(op->first);
    std::string second_str = Visit(op->second);

    return fmt::format("{}{}", first_str, second_str);
  }

  std::string ImplVisit(SpatialBlock op) override {
    std::vector<std::string> blocks;

    int num_bindings = (int)op->spatial_bindings.size();
    int num_blocks = (int)op->blocks.size();
    std::string ind = make_indent();
    ASSERT(num_bindings == num_blocks) << "The number of bindings and blocks mismatch.";
    increase_indent();
    for (int i = 0; i < num_bindings; ++i) {
      std::string binding = Visit(op->spatial_bindings[i]);
      std::string block = Visit(op->blocks[i]);
      blocks.push_back(fmt::format("{}spatial_{} {\n{}}\n", ind, binding, block));
    }
    decrease_indent();

    return fmt::format("{}", fmt::join(blocks, ""));
  }

  std::string ImplVisit(AtomBlock op) override {
    std::string ind = make_indent();
    if (op->isNullBlock()) {
      return ind + "\n";
    } else {
      return fmt::format("{}{}\n", ind, Visit(op->getStmt()));
    }
  }

  std::string ImplVisit(ReMapBlock op) override {
    std::string ind = make_indent();
    increase_indent();
    std::string body_str = Visit(op->body);
    decrease_indent();
    std::vector<std::string> mapping;
    for (auto m : op->mappings) {
      mapping.push_back(Visit(m));
    }
    return fmt::format("{}remap({}) {\n{}}\n", ind, fmt::join(mapping, ", "), body_str);
  }

  std::string ImplVisit(NdAllocBlock op) override {
    std::string ind = make_indent();
    increase_indent();
    std::string body_str = Visit(op->body);
    decrease_indent();
    std::vector<std::string> shape;
    for (auto s : op->shape) {
      shape.push_back(Visit(s));
    }
    return fmt::format("{}alloc_n({}[{}]@{}) {\n{}}\n", ind, Visit(op->var), fmt::join(shape, ", "),
                       Visit(op->memory_scope), body_str);
  }

  std::string ImplVisit(AllocBlock op) override {
    std::string ind = make_indent();
    increase_indent();
    std::string body_str = Visit(op->body);
    decrease_indent();

    return fmt::format("{}alloc({}[{}]@{}) {\n{}}\n", ind, Visit(op->var), Visit(op->length),
                       Visit(op->memory_scope), body_str);
  }

  void increase_indent() { indent_ += 1; }

  void decrease_indent() { indent_ -= 1; }

  std::string make_indent() {
    std::string ret = "";
    for (int i = 0; i < indent_; ++i) {
      ret += "  ";
    }
    return std::move(ret);
  }

 private:
  int indent_{0};
};

std::ostream& operator<<(std::ostream& os, IRBase ir) { return os << IRPrinter().Visit(ir); }

std::string repr(IRBase ir) { return IRPrinter().Visit(ir); }

}  // namespace domino
