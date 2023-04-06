#include <expr.h>

namespace domino {

Expr operator+(const Expr& a, const Expr& b) { return Add::make(a, b); }

Expr operator-(const Expr& a, const Expr& b) { return Sub::make(a, b); }

Expr operator*(const Expr& a, const Expr& b) { return Mul::make(a, b); }

Expr operator/(const Expr& a, const Expr& b) { return Div::make(a, b); }

Expr operator%(const Expr& a, const Expr& b) { return Mod::make(a, b); }

Expr operator-(const Expr& a) { return Neg::make(a); }

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
      return "spatial";
      break;
    case IterTypeKind::kReduce:
      return "reduce";
      break;
    case IterTypeKind::kUnroll:
      return "unroll";
      break;
    case IterTypeKind::kZigzag:
      return "zigzag";
      break;
    case IterTypeKind::kTensorized:
      return "tensorized";
      break;
    case IterTypeKind::kHybrid:
      return "hybrid";
      break;
    default:
      throw std::runtime_error(fmt::format("IterType not known: {}", int(kind)));
  }
}

Var var(const std::string dtype, const std::string& name) {
  DType type = DType::make(dtype);
  return Var::make(type, name);
}

ConstVar const_var(const std::string dtype, const std::string& name) {
  DType type = DType::make(dtype);
  return ConstVar::make(type, name);
}

}  // namespace domino