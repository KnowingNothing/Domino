#ifndef DOMINO_GENERAL_FUNCTOR_H
#define DOMINO_GENERAL_FUNCTOR_H

#include <ref.h>

#include <functional>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <unordered_map>

namespace domino {

namespace detail {

template <typename BasePtr, typename Func>
class default_vtable {
 public:
  template <typename T>
  inline void Set(Func f) {
    data_[std::type_index(typeid(T))] = f;
  }
  inline Func Get(BasePtr base) { return data_[std::type_index(typeid(*base))]; }

 private:
  std::unordered_map<std::type_index, Func> data_;
};

template <typename Base>
class default_pointer {
 public:
  using type = Ref<Base>;
  template <typename Derived>
  static inline Ref<Derived> cast(Ref<Base> base) {
    return base.template as<Derived>();
  }
};

}  // namespace detail

template <typename Visitor, typename Base, typename Deriveds, typename Func,
          typename = detail::default_pointer<Base>,
          template <typename, typename> typename = detail::default_vtable>
class GeneralFunctor;

template <typename Visitor, typename Base, typename... Deriveds, typename Pointer,
          template <typename, typename> typename Vtable, typename R, typename... Args>
class GeneralFunctor<Visitor, Base, std::tuple<void, Deriveds...>, R(Args...), Pointer, Vtable> {
  using BasePtr = typename Pointer::type;
  using VtableType = Vtable<BasePtr, R (*)(Visitor*, BasePtr, Args...)>;

 public:
  R Visit(BasePtr base, Args... args) {
    static VtableType vtable = BuildVtable();

    return vtable.Get(base)(static_cast<Visitor*>(this), base, std::forward<Args>(args)...);
  }

  R operator()(BasePtr base, Args... args) { return Visit(base, std::forward<Args>(args)...); }

 private:
  template <typename Derived, typename... Rest>
  static void Register(VtableType& vtable) {
    vtable.template Set<Derived>([](Visitor* visitor, BasePtr base, Args... args) -> R {
      return visitor->ImplVisit(Pointer::template cast<Derived>(base), std::forward<Args>(args)...);
    });
    if constexpr (sizeof...(Rest) > 0) {
      Register<Rest...>(vtable);
    }
  }

  static VtableType BuildVtable() {
    VtableType vtable;
    Register<Deriveds...>(vtable);
    return vtable;
  }
};

template <typename... Functors>
class MultiFunctors : public Functors... {
 public:
  using Functors::Visit...;
};

};  // namespace domino

#endif  // DOMINO_GENERAL_FUNCTOR_H
