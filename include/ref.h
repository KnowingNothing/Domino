#ifndef DOMINO_REF_H
#define DOMINO_REF_H

#include <cassert>
#include <iostream>
#include <memory>
#include <type_traits>

namespace domino {

template <typename T>
class Ref {
  template <class U>
  friend class Ref;

  std::shared_ptr<T> shr_ptr_;

  Ref(std::shared_ptr<T> sp) : shr_ptr_(sp) {}
  Ref(T* ptr) : shr_ptr_(ptr) {}

 public:
  Ref() = default;
  Ref(std::nullptr_t) : Ref() {}

  Ref(const Ref&) = default;
  Ref(Ref&&) = default;

  template <typename U, typename = std::enable_if_t<std::is_base_of_v<T, U>>>
  Ref(const Ref<U>& rhs) : shr_ptr_(std::static_pointer_cast<T>(rhs.shr_ptr_)) {}

  Ref& operator=(const Ref& rhs) = default;
  Ref& operator=(Ref&& rhs) = default;

  template <typename U, typename = std::enable_if_t<std::is_base_of_v<T, U>>>
  Ref& operator=(const Ref<U>& rhs) {
    shr_ptr_ = std::static_pointer_cast<T>(rhs.shr_ptr_);
    return *this;
  }

  template <class U>
  Ref<U> as() const {
    return std::dynamic_pointer_cast<U>(shr_ptr_);
  }

  bool defined() const { return shr_ptr_ != nullptr; }

  T& operator*() const {
    assert(defined());
    return *shr_ptr_;
  }

  T* operator->() const {
    assert(defined());
    return shr_ptr_.get();
  }

  T* get() const { return shr_ptr_.get(); }

  friend bool operator==(const Ref& lhs, const Ref& rhs) { return lhs.shr_ptr_ == rhs.shr_ptr_; }
  friend bool operator!=(const Ref& lhs, const Ref& rhs) { return !(lhs == rhs); }

  template <class... Args>
  static Ref make(Args&&... args) {
    return Ref(std::make_shared<T>(std::forward<Args>(args)...));
  }
};

};  // namespace domino

#endif
