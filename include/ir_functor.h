#ifndef DOMINO_IR_FUNCTOR_H
#define DOMINO_IR_FUNCTOR_H

#include <arch.h>
#include <block.h>
#include <expr.h>
#include <general_functor.h>
#include <ir_base.h>
#include <kernel.h>
#include <ref.h>
#include <stmt.h>

#include <tuple>

namespace domino {

using namespace domino::arch;

template <typename F, bool Cache = true>
class IRFunctor;
template <typename R, typename... Args, bool Cache>
class IRFunctor<R(Args...), Cache> : public GeneralFunctor<IRFunctor<R(Args...), Cache>, IRBaseNode,
                                                           std::tuple<void
#define X_DECL_IR(X) , X##Node
#include <x_macro/ir.x.h>
                                                                      >,
                                                           R(Args...), Cache> {
 public:
#define X_DECL_IR(X) \
  virtual R ImplVisit(X, Args...) { throw std::runtime_error("not implemented"); }
#include <x_macro/ir.x.h>
};

}  // namespace domino

#endif
