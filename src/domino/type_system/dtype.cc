#include <type_system/dtype.h>

namespace domino {

std::string dtype_kind_to_string(DTypeKind kind) {
  switch (kind) {
    case DTypeKind::kInt:
      return "int";
      break;
    case DTypeKind::kUInt:
      return "uint";
      break;
    case DTypeKind::kFloat:
      return "float";
      break;
    case DTypeKind::kBFloat:
      return "bfloat";
      break;
    case DTypeKind::kTFloat:
      return "tfloat";
      break;
    case DTypeKind::kMemRef:
      return "mem_ref";
      break;
    case DTypeKind::kString:
      return "string";
      break;
    case DTypeKind::kComplex:
      return "complex";
      break;
    case DTypeKind::kIGNORE:
      return "ignore";
      break;
    case DTypeKind::kUNKNOWN:
      return "unknown";
      break;
    default:
      throw std::runtime_error(fmt::format("Unsupported data type: {}", int(kind)));
  }
}
}  // namespace domino