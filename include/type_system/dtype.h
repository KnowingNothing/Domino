#ifndef DOMINO_DTYPE_H
#define DOMINO_DTYPE_H

#include <logging/logging.h>

namespace domino {

using namespace logging;

enum class DTypeKind : int {
    kInt = 0,
    kUInt= 1,
    kFloat = 2,
    kBFloat = 3,
    kTFloat = 4,
    kMemRef = 5,
    kString = 6,
    kComplex = 7,
    kIGNORE = 254,  /// there is no need for type
    kUNKNOWN = 255  /// type is needed, but not known
};

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

        bool is_int() const {
            return type_kind == DTypeKind::kInt;
        }

        bool is_uint() const {
            return type_kind == DTypeKind::kUInt;
        }

        bool is_float() const {
            return type_kind == DTypeKind::kFloat;
        }

        bool is_bfloat() const {
            return type_kind == DTypeKind::kBFloat;
        }

        bool is_tfloat() const {
            return type_kind == DTypeKind::kTFloat;
        }

        bool is_memref() const {
            return type_kind == DTypeKind::kMemRef;
        }

        bool is_string() const {
            return type_kind == DTypeKind::kString;
        }

        bool is_complex() const {
            return type_kind == DTypeKind::kComplex;
        }

        bool is_ignore() const {
            return type_kind == DTypeKind::kIGNORE;
        }

        bool is_unknown() const {
            return type_kind == DTypeKind::kUNKNOWN;
        }

        DType copy() const {
            return DType(type_kind, bits, lane);
        }

        DType with_lanes(int lanes) const {
            return DType(type_kind, bits, lanes);
        }

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

    DTypeKind type_kind;
    int bits;
    int lane;
};

}  // namespace domino

#endif // DOMINO_DTYPE_H