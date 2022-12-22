#include <expr.h>

namespace domino {

Expr operator+(const Expr& a, const Expr& b) { return Add::make(a, b); }

Expr operator-(const Expr& a, const Expr& b) { return Sub::make(a, b); }

Expr operator*(const Expr& a, const Expr& b) { return Mul::make(a, b); }

Expr operator/(const Expr& a, const Expr& b) { return Div::make(a, b); }

Expr operator%(const Expr& a, const Expr& b) { return Mod::make(a, b); }

ConstInt const_int(long long int value, int bits, int lanes) {
  return ConstInt::make(value, bits, lanes);
}

ConstUInt const_uint(unsigned long long int value, int bits, int lanes) {
  return ConstUInt::make(value, bits, lanes);
}

ConstFloat const_float(double value, int bits, int lanes) {
  return ConstFloat::make(value, bits, lanes);
}

ConstString const_string(std::string value) { return ConstString::make(value); }

std::string iter_type_to_string(IterTypeKind kind) {
  switch (kind) {
    case IterTypeKind::kSpatial:
      return "spatial"; break;
    case IterTypeKind::kReduce:
      return "reduce"; break;
    default:
      throw std::runtime_error(fmt::format("IterType not known: {}", int(kind)));
  }
}

}  // namespace domino