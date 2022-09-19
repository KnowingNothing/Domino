#ifndef DOMINO_IR_FUNCTOR_H
#define DOMINO_IR_FUNCTOR_H

#include <expr.h>
#include <general_functor.h>
#include <ir_base.h>
#include <ref.h>

#include <tuple>

namespace domino {

template <typename F>
class IRFunctor;
template <typename R, typename... Args>
class IRFunctor<R(Args...)> : public GeneralFunctor<IRFunctor<R(Args...)>, IRBaseNode,
                                                    std::tuple<void
#define X_DECL_IR(X) , X##Node
#include <x_macro/ir.x.h>
                                                               >,
                                                    R(Args...)> {
 public:
#define X_DECL_IR(X) \
  virtual R ImplVisit(X, Args...) { throw std::runtime_error("not implemented"); }
#include <x_macro/ir.x.h>
};

}  // namespace domino

#endif
