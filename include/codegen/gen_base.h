#ifndef DOMINO_CODEGEN_GEN_BASE_H
#define DOMINO_CODEGEN_GEN_BASE_H

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <ir_base.h>
#include <ir_functor.h>

namespace domino {

namespace codegen {

class CodeGenBase : public IRFunctor<std::string()> {
 protected:
  /// expressions
  std::string ImplVisit(MemRef mem_ref) override {
    return fmt::format("({}+{})", Visit(mem_ref->var), Visit(mem_ref->offset));
  }

  std::string ImplVisit(ValueRef value_ref) override {
    return fmt::format("(&{})", Visit(value_ref->var));
  }

  std::string ImplVisit(ArrayRef array_ref) override {
    return fmt::format("(&{}{})", Visit(array_ref->var), PrintNDimIndices(array_ref->args));
  }

  std::string visit_bin_op(BinExpr bin, std::string op) {
    return fmt::format("({} {} {})", Visit(bin->a), op, Visit(bin->b));
  }

  std::string ImplVisit(Add op) override { return this->visit_bin_op(op, "+"); }

  std::string ImplVisit(Sub op) override { return this->visit_bin_op(op, "-"); }

  std::string ImplVisit(Mul op) override { return this->visit_bin_op(op, "*"); }

  std::string ImplVisit(Div op) override { return this->visit_bin_op(op, "/"); }

  std::string ImplVisit(Mod op) override { return this->visit_bin_op(op, "%"); }

  std::string ImplVisit(FloorDiv op) override { return this->visit_bin_op(op, "/"); }

  // std::string ImplVisit(FloorMod op) override not implemented

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
    return fmt::format("(({}){})", std::string(op->dtype), Visit(op->a));
  }

  // std::string ImplVisit(Broadcast op) override not implemented

  std::string ImplVisit(Neg op) override { return fmt::format("(-{})", Visit(op->a)); }

  std::string ImplVisit(Not op) override { return fmt::format("(!{})", Visit(op->a)); }

  std::string ImplVisit(BitNot op) override { return fmt::format("(~{})", Visit(op->a)); }

  // std::string ImplVisit(Ceil op) override not implemented

  // std::string ImplVisit(Floor op) override not implemented

  std::string ImplVisit(Select op) override {
    return fmt::format("({} ? {} : {})", Visit(op->a), Visit(op->b), Visit(op->c));
  }

  // std::string ImplVisit(Range op) override not implemented

  std::string ImplVisit(ExprList op) override {
    std::vector<std::string> operands;
    for (auto v : op->value_list) {
      operands.push_back(Visit(v));
    }
    return fmt::format("{}", fmt::join(operands, ", "));
  }

  std::string PrintNDimIndices(ExprList op) {
    std::vector<std::string> operands;
    for (auto v : op->value_list) {
      operands.push_back(Visit(v));
    }
    return fmt::format("[{}]", fmt::join(operands, "]["));
  }

  std::string ImplVisit(CondAll op) override {
    std::vector<std::string> operands;
    for (auto v : op->phases->value_list) {
      operands.push_back(Visit(v));
    }
    return fmt::format("({})", fmt::join(operands, " && "));
  }

  std::string ImplVisit(CondAny op) override {
    std::vector<std::string> operands;
    for (auto v : op->phases->value_list) {
      operands.push_back(Visit(v));
    }
    return fmt::format("({})", fmt::join(operands, " || "));
  }

  std::string ImplVisit(ConstInt op) override { return fmt::format("{}", op->value); }

  std::string ImplVisit(ConstUInt op) override { return fmt::format("{}", op->value); }

  std::string ImplVisit(ConstFloat op) override { return fmt::format("{}", op->value); }

  // std::string ImplVisit(ConstBFloat op) override not implemented

  // std::string ImplVisit(ConstTFloat op) override not implemented

  std::string ImplVisit(ConstString op) override { return fmt::format("{}", op->value); }

  std::string ImplVisit(Var op) override { return fmt::format("{}", Visit(op->id)); }

  // std::string ImplVisit(Iterator op) override not implemented

  std::string ImplVisit(NdLoad op) override {
    return fmt::format("({}{})", Visit(op->mem_ref), PrintNDimIndices(op->indices));
  }

  std::string ImplVisit(Load op) override {
    return fmt::format("({}[{}])", Visit(op->mem_ref), Visit(op->addr));
  }

  std::string ImplVisit(MapVar op) override {
    return fmt::format("({} {} = {})", std::string(op->var->dtype), Visit(op->var),
                       Visit(op->expr));
  }

  // std::string ImplVisit(Slice op) override not implemented

  // std::string ImplVisit(MemSlice op) override not implemented

  std::string ImplVisit(Call op) override {
    return fmt::format("({}({}))", Visit(op->func), Visit(op->args));
  }

  // std::string ImplVisit(PackValue op) override not implemented

  /// statements
  std::string ImplVisit(NdStore op) override {
    return fmt::format("{}{} = {};", Visit(op->mem_ref), PrintNDimIndices(op->indices), Visit(op->value));
  }

  std::string ImplVisit(Store op) override {
    return fmt::format("{}[{}] = {};", Visit(op->mem_ref), Visit(op->addr), Visit(op->value));
  }

  std::string ImplVisit(Evaluate op) override { return fmt::format("({});", Visit(op->expr)); }

  /// blocks
  // std::string ImplVisit(AttrBlock op) override not implemented

  // std::string ImplVisit(NdForBlock op) override not implemented

  std::string ImplVisit(ForBlock op) override {
    /// increase indent
    increase_indent();
    std::string body_str = Visit(op->body);
    /// decrease indent
    decrease_indent();
    std::string init = fmt::format("{} {}={}", std::string(op->iter->var->dtype),
                                   Visit(op->iter->var), Visit(op->iter->range->beg));
    std::string cond = fmt::format("{}<{}", Visit(op->iter->var), Visit(op->iter->range->extent));
    std::string step = fmt::format("{}+={}", Visit(op->iter->var), Visit(op->iter->range->step));
    std::string ind = make_indent();

    std::string left = "{";
    std::string right = "}";
    return fmt::format("{}for({}; {}; {}) {}\n{}{}{}\n", ind, init, cond, step, left, body_str, ind,
                       right);
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

  // std::string ImplVisit(SpatialBlock op) override not implemented

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
    std::string body_str = Visit(op->body);
    std::vector<std::string> mapping;
    for (auto m : op->mappings) {
      mapping.push_back(ind + Visit(m) + ";\n");
    }

    return fmt::format("{}{}", fmt::join(mapping, ""), body_str);
  }

  std::string ImplVisit(NdAllocBlock op) override {
    std::string ind = make_indent();
    std::string body_str = Visit(op->body);
    std::vector<std::string> shape;
    for (auto s : op->shape) {
      shape.push_back(Visit(s));
    }

    std::string left = "{";
    std::string right = "}";
    return fmt::format("{}{} {}[{}] = {}0{};\n{}", ind, std::string(op->var->dtype), Visit(op->var),
                       fmt::join(shape, ", "), left, right, body_str);
  }

  std::string ImplVisit(AllocBlock op) override {
    std::string ind = make_indent();
    std::string body_str = Visit(op->body);

    std::string left = "{";
    std::string right = "}";
    return fmt::format("{}{} {}[{}] = {}0{};\n{}", ind, std::string(op->var->dtype), Visit(op->var),
                       Visit(op->length), left, right, body_str);
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
  int indent_{1};
};

}  // namespace codegen

}  // namespace domino

#endif  // DOMINO_CODEGEN_GEN_BASE_H