#ifndef DOMINO_CODEGEN_GEN_TILEFLOW_H
#define DOMINO_CODEGEN_GEN_TILEFLOW_H

#include <codegen/gen_base.h>

#include <unordered_map>
#include <vector>

namespace domino {
using namespace domino::arch;
namespace codegen {

class CodeGenTileFlow : public CodeGenBase {
 protected:
  std::string ImplVisit(ComputeLevel op) override;
  std::string ImplVisit(MemoryLevel op) override;
  /// TileFlow only accepts single character as name
  std::string renaming(std::string name) {
    return name;
    // disable renaming
    // if (!renaming_table_.count(name)) {
    //   /// starts from 'K', leave 'A-J' to tensor names
    //   renaming_table_[name] = loop_renaming_index_;
    //   loop_renaming_index_ += 1;
    //   ASSERT(loop_renaming_index_ < 16) << "Too many loops!\n";
    // }
    // return char('K' + renaming_table_.at(name));
  }

 private:
  std::unordered_map<Arch, Arch> parent_pointers_;
  std::unordered_map<Arch, std::vector<Iterator>> loop_table_;
  int loop_renaming_index_{0};
  std::unordered_map<std::string, int> renaming_table_;
  bool top_{true};
};

std::string codegen_tileflow(IRBase tree);

}  // namespace codegen

}  // namespace domino

#endif  // DOMINO_CODEGEN_GEN_TILEFLOW_H