#ifndef DOMINO_DTYPE_H
#define DOMINO_DTYPE_H

#include <fmt/core.h>
#include <logging/logging.h>

namespace domino {

using namespace logging;

enum class DTypeKind : int {
  kInt = 0,
  kUInt = 1,
  kFloat = 2,
  kBFloat = 3,
  kTFloat = 4,
  kMemRef = 5,
  kString = 6,
  kComplex = 7,
  kIGNORE = 254,  /// there is no need for type
  kUNKNOWN = 255  /// type is needed, but not known
};

std::string dtype_kind_to_string(DTypeKind kind);

/**
 * \brief DType class for scalar data type with vectorize support.
 *
 */
class DType {
 public:
  DType() = delete;
  DType(DTypeKind kind, int bi, int la) : type_kind(kind), bits(bi), lane(la) {}
  DType(const DType& others) = default;
  DType(DType&& others) = default;
  ~DType() = default;

  bool is_int() const { return type_kind == DTypeKind::kInt; }

  bool is_uint() const { return type_kind == DTypeKind::kUInt; }

  bool is_float() const { return type_kind == DTypeKind::kFloat; }

  bool is_bfloat() const { return type_kind == DTypeKind::kBFloat; }

  bool is_tfloat() const { return type_kind == DTypeKind::kTFloat; }

  bool is_memref() const { return type_kind == DTypeKind::kMemRef; }

  bool is_string() const { return type_kind == DTypeKind::kString; }

  bool is_complex() const { return type_kind == DTypeKind::kComplex; }

  bool is_ignore() const { return type_kind == DTypeKind::kIGNORE; }

  bool is_unknown() const { return type_kind == DTypeKind::kUNKNOWN; }

  DType copy() const { return DType(type_kind, bits, lane); }

  DType with_lanes(int lanes) const { return DType(type_kind, bits, lanes); }

  long long max_limit() const {
    ASSERT(lane == 1) << "Can't get the max_limit value for vectorized type.";
    ASSERT(is_int() || is_uint()) << "Only support get max_limit for int/uint.";
    long long ret = -1;
    if (is_int()) {
      ret = (1L << (bits - 1)) - 1;
    } else if (is_uint()) {
      ret = (1 << bits) - 1;
    }
    ASSERT(ret > 0) << "Value overflow.";
    return ret;
  }

  long long min_limit() const {
    ASSERT(lane == 1) << "Can't get the min_limit value for vectorized type.";
    ASSERT(is_int() || is_uint()) << "Only support get min_limit for int/uint.";
    long long ret = 1;
    if (is_int()) {
      ret = -(1L << (bits - 1));
    } else if (is_uint()) {
      ret = 0;
    }
    ASSERT(ret <= 0) << "Value overflow.";
    return ret;
  }

  static DType make(const std::string& dtype_str) {
    int length = (int)dtype_str.size();
    int split_pos = 0;
    DTypeKind kind = DTypeKind::kUNKNOWN;
    int default_bits = 0;
    if (dtype_str.substr(0, 3) == std::string("int")) {
      split_pos = 3;
      kind = DTypeKind::kInt;
      default_bits = 32;
    } else if (dtype_str.substr(0, 4) == std::string("uint")) {
      split_pos = 4;
      kind = DTypeKind::kUInt;
      default_bits = 32;
    } else if (dtype_str.substr(0, 5) == std::string("float")) {
      split_pos = 5;
      kind = DTypeKind::kFloat;
      default_bits = 32;
    } else if (dtype_str.substr(0, 6) == std::string("bfloat")) {
      split_pos = 6;
      kind = DTypeKind::kBFloat;
      default_bits = 16;
    } else if (dtype_str.substr(0, 6) == std::string("tfloat")) {
      split_pos = 6;
      kind = DTypeKind::kTFloat;
      default_bits = 32;
    } else if (dtype_str.substr(0, 7) == std::string("mem_ref")) {
      split_pos = 7;
      kind = DTypeKind::kMemRef;
      default_bits = 0;
    } else if (dtype_str.substr(0, 6) == std::string("string")) {
      split_pos = 6;
      kind = DTypeKind::kString;
      default_bits = 0;
    } else if (dtype_str.substr(0, 4) == std::string("bool")) {
      split_pos = 4;
      kind = DTypeKind::kUInt;
      default_bits = 1;
    } else if (dtype_str.substr(0, 7) == std::string("complex")) {
      split_pos = 7;
      kind = DTypeKind::kComplex;
      default_bits = 64;
    } else if (dtype_str.substr(0, 6) == std::string("ignore")) {
      split_pos = 6;
      kind = DTypeKind::kIGNORE;
      default_bits = 0;
    } else if (dtype_str.substr(0, 7) == std::string("unknown")) {
      split_pos = 7;
      kind = DTypeKind::kUNKNOWN;
      default_bits = 0;
    } else {
      std::string message = std::string(fmt::format("Cant't parse type string {}.", dtype_str));
      throw std::runtime_error(message);
    }

    std::string suffix = dtype_str.substr(split_pos);
    split_pos = suffix.find("x");
    std::string bits_str = "", lanes_str = "";
    int bits_num = default_bits, lanes_num = 1;
    if (split_pos != std::string::npos) {
      bits_str = suffix.substr(0, split_pos);
      lanes_str = suffix.substr(split_pos + 1);
    } else {
      bits_str = suffix;
    }

    if (bits_str.size() > 0) {
      bits_num = stoi(bits_str);
    }
    if (lanes_str.size() > 0) {
      lanes_num = stoi(lanes_str);
    }

    return DType(kind, bits_num, lanes_num);
  }

  operator std::string() const {
    if (lane == 1) {
      return fmt::format("{}{}", dtype_kind_to_string(type_kind), bits);
    } else {
      return fmt::format("{}{}x{}", dtype_kind_to_string(type_kind), bits, lane);
    }
  }

  bool operator==(const DType& other) const {
    return (type_kind == other.type_kind) && (bits == other.bits) && (lane == other.lane);
  }

  bool operator!=(const DType& other) const { return !((*this) == other); }

  DTypeKind type_kind;
  int bits;
  int lane;
};

}  // namespace domino

#endif  // DOMINO_DTYPE_H