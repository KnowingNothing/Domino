#ifndef DOMINO_KERNEL_H
#define DOMINO_KERNEL_H

#include <arch.h>
#include <block.h>
#include <expr.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <ir_base.h>
#include <ref.h>
#include <stmt.h>

#include <vector>

namespace domino {

/**
 * \brief The kernel signature struct
 *
 * We assume all the kernel return type is void
 *
 * \param kernel_name the name of the kernel in string
 * \param kernel_args a list of Var, the arguments (pointers only)
 * TODO: we need to support scalars later
 */
class KernelSignatureNode : public IRBaseNode {
 public:
  std::string kernel_name;
  std::vector<Var> kernel_args;
  std::vector<Var> scalar_args;

  KernelSignatureNode(std::string name, std::vector<Var> args, std::vector<Var> scalars)
      : kernel_name(std::move(name)),
        kernel_args(std::move(args)),
        scalar_args(std::move(scalars)) {}
};

using KernelSignature = Ref<KernelSignatureNode>;

/**
 * \brief Kernel structure
 *
 * \param signature kernel signature
 * \param body kernel body, a block
 * \param source kernel source code
 */
class KernelNode : public IRBaseNode {
 public:
  KernelSignature signature;
  Block body;
  std::string source;

  KernelNode(KernelSignature signature, Block body)
      : signature(std::move(signature)), body(std::move(body)) {}

  bool compiled() const { return this->source.size() > 0; }

  std::string genSignature() const {
    std::vector<std::string> args;
    for (auto arg : this->signature->kernel_args) {
      if (!arg->IsConst()) {
        args.push_back(fmt::format("{}* {}", std::string(arg->dtype), arg->id->value));
      } else {
        args.push_back(fmt::format("const {}* {}", std::string(arg->dtype), arg->id->value));
      }
    }
    for (auto arg : this->signature->scalar_args) {
      if (!arg->IsConst()) {
        args.push_back(fmt::format("{} {}", std::string(arg->dtype), arg->id->value));
      } else {
        args.push_back(fmt::format("const {} {}", std::string(arg->dtype), arg->id->value));
      }
    }
    return fmt::format("void {}({})", this->signature->kernel_name, fmt::join(args, ", "));
  }

  std::string genFunction() const {
    std::string signature = this->genSignature();
    std::string left = "{";
    std::string right = "}";
    return fmt::format("{} {}\n{}{}", signature, left, this->source, right);
  }
};

using Kernel = Ref<KernelNode>;

}  // namespace domino

#endif  // DOMINO_KERNEL_H