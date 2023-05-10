#ifndef DOMINO_CODEGEN_GEN_ARM_M_H
#define DOMINO_CODEGEN_GEN_ARM_M_H

#include <codegen/gen_c.h>
#include <logging/logging.h>
#include <type_system/dtype.h>

namespace domino {

namespace codegen {

class CodeGenARM_M : public CodeGenC {
 protected:
  std::string ImplVisit(PackValue op) override {
    ASSERT(op->value_list->value_list.size() == 2U) << "Only support pack two values";
    std::vector<std::string> values;
    for (auto v : op->value_list->value_list) {
      values.push_back(Visit(v));
    }
    if ((op->value_list->value_list[0]->dtype == "int16") &&
        (op->value_list->value_list[1]->dtype == "int16") && (op->dtype == "int32")) {
      return fmt::format("__PKHBT({}, {}, 16)", values[0], values[1]);
    } else {
      throw std::runtime_error("Unsupported PackValue pattern.");
    }
  }

  std::string ImplVisit(AttrBlock op) override {
    std::string key = op->key->value;
    if (key == "ring_buffer_check_bound") {
      Load load = op->value.as<LoadNode>();
      ASSERT(load.defined());
      MemRef mem_ref = load->mem_ref;
      Var v = mem_ref->var;
      Expr offset = mem_ref->offset;
      Expr addr = load->addr;
      std::string ind = make_indent();
      return fmt::format(
          "{}{} = (void*)({} + {}) > (void*)__RING_BUF_END__ ? ({}*)((unsigned char*){} - "
          "__RING_BUF_SIZE__) : {};\n{}",
          ind, Visit(v), Visit(mem_ref), Visit(addr), std::string(v->dtype), Visit(v), Visit(v),
          Visit(op->body));
    } else {
      throw std::runtime_error("Unsupported attribute " + key);
    }
  }
};

std::string codegen_arm_m(IRBase tree);

}  // namespace codegen

}  // namespace domino

#endif  // DOMINO_CODEGEN_GEN_C_H