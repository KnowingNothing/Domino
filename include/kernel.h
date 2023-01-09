#ifndef DOMINO_KERNEL_H
#define DOMINO_KERNEL_H

#include <arch.h>
#include <block.h>
#include <expr.h>
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
 * \param kernel_args a list of Expr, the arguments
 */
class KernelSignatureNode : public IRBaseNode {
 public:
  std::string kernel_name;
  std::vector<Var> kernel_args;

  KernelSignatureNode(std::string name, std::vector<Var> args)
      : kernel_name(std::move(name)), kernel_args(std::move(kernel_args)) {}
};

using KernelSignature = Ref<KernelSignatureNode>;

/**
 * \brief Kernel structure
 *
 * \param signature kernel signature
 * \param body kernel body, a block
 */
class KernelNode : public IRBaseNode {
 public:
  KernelSignature signature;
  Block body;

  KernelNode(KernelSignature signature, Block body) : signature(signature), body(body) {}
};

using Kernel = Ref<KernelNode>;

}  // namespace domino

#endif  // DOMINO_KERNEL_H