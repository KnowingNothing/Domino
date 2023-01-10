#ifndef DOMINO_EXPR_H
#define DOMINO_EXPR_H

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <ir_base.h>
#include <logging/logging.h>
#include <ref.h>
#include <type_system/dtype.h>

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace domino {
using namespace logging;
/// Don't use X_Macro for reference declaration
/// for better debug experience

// #define X_DECL_EXPR(X) \
//   class X##Node;       \
//   using X = Ref<X##Node>;
// #include <x_macro/expr.x.h>

/**
 * \brief The base class for all the expressions
 *
 * \param dtype the data type
 */
class ExprNode : public IRBaseNode {
 public:
  ExprNode(DType d) : dtype(std::move(d)) {}
  ExprNode(const ExprNode&) = default;
  ExprNode(ExprNode&&) = default;
  virtual bool IsConst() const { return false; }
  virtual operator std::string() const { return fmt::format("Expr({})", std::string(dtype)); }

  DType dtype;
};

using Expr = Ref<ExprNode>;

///=----------------------------------------------------------------------------=///
///
/// Non-Terminal IR Node
///
///=----------------------------------------------------------------------------=///

/**
 * \brief Binary operation expr
 *
 * \param a lhs expr
 * \param b rhs expr
 */
struct BinExprNode : public ExprNode {
 public:
  Expr a, b;

  BinExprNode(DType dtype, Expr lhs, Expr rhs)
      : ExprNode(std::move(dtype)), a(std::move(lhs)), b(std::move(rhs)) {
    ASSERT(a.defined() && b.defined());
    ASSERT(a->dtype == b->dtype) << fmt::format(
        "Binary expression expects the same type for operands, but get {} and {}",
        std::string(a->dtype), std::string(b->dtype));
  }

  operator std::string() const override {
    return fmt::format("BinExpr({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using BinExpr = Ref<BinExprNode>;

/**
 * \brief Unary operation expr
 *
 * \param a operand expr
 */
class UniExprNode : public ExprNode {
 public:
  Expr a;

  UniExprNode(DType dtype, Expr opr) : ExprNode(std::move(dtype)), a(std::move(opr)) {
    ASSERT(a.defined());
  }

  operator std::string() const override {
    return fmt::format("UniExpr({}, {})", std::string(this->dtype), std::string(this->a));
  }
};

using UniExpr = Ref<UniExprNode>;

/**
 * \brief Ternary operation expr
 *
 * \param a operand a
 * \param b operand b
 * \param c operand c
 */
class TerExprNode : public ExprNode {
 public:
  Expr a, b, c;

  TerExprNode(DType dtype, Expr a, Expr b, Expr c)
      : ExprNode(std::move(dtype)), a(std::move(a)), b(std::move(b)), c(std::move(c)) {
    ASSERT(a.defined() && b.defined() && c.defined());
  }

  operator std::string() const override {
    return fmt::format("TerExpr({}, {}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b), std::string(this->c));
  }
};

using TerExpr = Ref<TerExprNode>;

/**
 * \brief Constant expression
 *
 */
class ConstExprNode : public ExprNode {
 public:
  ConstExprNode(DType dtype) : ExprNode(std::move(dtype)) {}
  bool IsConst() const override { return true; }

  operator std::string() const override {
    return fmt::format("ConstExpr({})", std::string(this->dtype));
  }
};

using ConstExpr = Ref<ConstExprNode>;

/**
 * \brief Mutable expression
 */
class MutableExprNode : public ExprNode {
 public:
  MutableExprNode(DType dtype) : ExprNode(std::move(dtype)) {}

  operator std::string() const override {
    return fmt::format("MutableExpr({})", std::string(this->dtype));
  }
};

using MutableExpr = Ref<MutableExprNode>;

/// forward decl
class VarNode;
using Var = Ref<VarNode>;

/**
 * \brief Memory reference expression.
 *
 * This is used to express a data array pointer with offset to an original array head.
 *
 * \param var the array head variable
 * \param offset the offset value
 */
class MemRefNode : public ExprNode {
 public:
  MemRefNode(Var v, Expr offset) : ExprNode(DType::make("mem_ref")), var(v), offset(offset) {
    ASSERT(v.defined() && offset.defined());
  }

  operator std::string() const override {
    return fmt::format("MemRef({}, {})", std::string(this->var), std::string(this->offset));
  }

  Var var;
  Expr offset;
};

using MemRef = Ref<MemRefNode>;

///=----------------------------------------------------------------------------=///
///
/// Binary Operation IR Node
///
///=----------------------------------------------------------------------------=///

class AddNode : public BinExprNode {
 public:
  AddNode(Expr a, Expr b) : BinExprNode(a->dtype, a, b) {}

  operator std::string() const override {
    return fmt::format("Add({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using Add = Ref<AddNode>;

class SubNode : public BinExprNode {
 public:
  SubNode(Expr a, Expr b) : BinExprNode(a->dtype, a, b) {}

  operator std::string() const override {
    return fmt::format("Sub({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using Sub = Ref<SubNode>;

class MulNode : public BinExprNode {
 public:
  MulNode(Expr a, Expr b) : BinExprNode(a->dtype, a, b) {}

  operator std::string() const override {
    return fmt::format("Mul({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using Mul = Ref<MulNode>;

class DivNode : public BinExprNode {
 public:
  DivNode(Expr a, Expr b) : BinExprNode(a->dtype, a, b) {}

  operator std::string() const override {
    return fmt::format("Div({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using Div = Ref<DivNode>;

class ModNode : public BinExprNode {
 public:
  ModNode(Expr a, Expr b) : BinExprNode(a->dtype, a, b) {}

  operator std::string() const override {
    return fmt::format("Mod({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using Mod = Ref<ModNode>;

class FloorDivNode : public BinExprNode {
 public:
  FloorDivNode(Expr a, Expr b) : BinExprNode(a->dtype, a, b) {}

  operator std::string() const override {
    return fmt::format("FloorDiv({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using FloorDiv = Ref<FloorDivNode>;

class FloorModNode : public BinExprNode {
 public:
  FloorModNode(Expr a, Expr b) : BinExprNode(a->dtype, a, b) {}

  operator std::string() const override {
    return fmt::format("FloorMod({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using FloorMod = Ref<FloorModNode>;

class AndNode : public BinExprNode {
 public:
  AndNode(Expr a, Expr b) : BinExprNode(DType::make("bool"), a, b) {}

  operator std::string() const override {
    return fmt::format("And({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using And = Ref<AndNode>;

class OrNode : public BinExprNode {
 public:
  OrNode(Expr a, Expr b) : BinExprNode(DType::make("bool"), a, b) {}

  operator std::string() const override {
    return fmt::format("Or({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using Or = Ref<OrNode>;

class XOrNode : public BinExprNode {
 public:
  XOrNode(Expr a, Expr b) : BinExprNode(DType::make("bool"), a, b) {}

  operator std::string() const override {
    return fmt::format("XOr({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using XOr = Ref<XOrNode>;

class BitAndNode : public BinExprNode {
 public:
  BitAndNode(Expr a, Expr b) : BinExprNode(a->dtype, a, b) {}

  operator std::string() const override {
    return fmt::format("BitAnd({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using BitAnd = Ref<BitAndNode>;

class BitOrNode : public BinExprNode {
 public:
  BitOrNode(Expr a, Expr b) : BinExprNode(a->dtype, a, b) {}

  operator std::string() const override {
    return fmt::format("BitOr({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using BitOr = Ref<BitOrNode>;

class BitXOrNode : public BinExprNode {
 public:
  BitXOrNode(Expr a, Expr b) : BinExprNode(a->dtype, a, b) {}

  operator std::string() const override {
    return fmt::format("BitXOr({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using BitXOr = Ref<BitXOrNode>;

class GTNode : public BinExprNode {
 public:
  GTNode(Expr a, Expr b) : BinExprNode(DType::make("bool"), a, b) {}

  operator std::string() const override {
    return fmt::format("GT({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using GT = Ref<GTNode>;

class GENode : public BinExprNode {
 public:
  GENode(Expr a, Expr b) : BinExprNode(DType::make("bool"), a, b) {}

  operator std::string() const override {
    return fmt::format("GE({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using GE = Ref<GENode>;

class LTNode : public BinExprNode {
 public:
  LTNode(Expr a, Expr b) : BinExprNode(DType::make("bool"), a, b) {}

  operator std::string() const override {
    return fmt::format("LT({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using LT = Ref<LTNode>;

class LENode : public BinExprNode {
 public:
  LENode(Expr a, Expr b) : BinExprNode(DType::make("bool"), a, b) {}

  operator std::string() const override {
    return fmt::format("LE({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using LE = Ref<LENode>;

class EQNode : public BinExprNode {
 public:
  EQNode(Expr a, Expr b) : BinExprNode(DType::make("bool"), a, b) {}

  operator std::string() const override {
    return fmt::format("EQ({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using EQ = Ref<EQNode>;

class NENode : public BinExprNode {
 public:
  NENode(Expr a, Expr b) : BinExprNode(DType::make("bool"), a, b) {}

  operator std::string() const override {
    return fmt::format("NE({}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b));
  }
};

using NE = Ref<NENode>;

///=----------------------------------------------------------------------------=///
///
/// Unary Operation IR Node
///
///=----------------------------------------------------------------------------=///
class CastNode : public UniExprNode {
 public:
  CastNode(DType dtype, Expr a) : UniExprNode(dtype, a) {}

  operator std::string() const override {
    return fmt::format("Cast({}, {})", std::string(this->dtype), std::string(this->a));
  }
};

using Cast = Ref<CastNode>;

class BroadcastNode : public UniExprNode {
 public:
  BroadcastNode(Expr a, int lane) : UniExprNode(a->dtype.with_lanes(lane), a) {}

  BroadcastNode(DType dtype, Expr a) : UniExprNode(dtype, a) {}

  operator std::string() const override {
    return fmt::format("Broadcast({}, {})", std::string(this->dtype), std::string(this->a));
  }
};

using Broadcast = Ref<BroadcastNode>;

class NegNode : public UniExprNode {
 public:
  NegNode(Expr a) : UniExprNode(a->dtype, a) {}

  operator std::string() const override {
    return fmt::format("Neg({}, {})", std::string(this->dtype), std::string(this->a));
  }
};

using Neg = Ref<NegNode>;

class NotNode : public UniExprNode {
 public:
  NotNode(Expr a) : UniExprNode(DType::make("bool"), a) {}

  operator std::string() const override {
    return fmt::format("Not({}, {})", std::string(this->dtype), std::string(this->a));
  }
};

using Not = Ref<NotNode>;

class BitNotNode : public UniExprNode {
 public:
  BitNotNode(Expr a) : UniExprNode(a->dtype, a) {}

  operator std::string() const override {
    return fmt::format("BitNot({}, {})", std::string(this->dtype), std::string(this->a));
  }
};

using BitNot = Ref<BitNotNode>;

class CeilNode : public UniExprNode {
 public:
  CeilNode(DType dtype, Expr a) : UniExprNode(dtype, a) {
    ASSERT(dtype.is_int() || dtype.is_uint())
        << fmt::format("Ceil output dtype should be Int or UInt, but get {}", std::string(dtype));
  }

  operator std::string() const override {
    return fmt::format("Ceil({}, {})", std::string(this->dtype), std::string(this->a));
  }
};

using Ceil = Ref<CeilNode>;

class FloorNode : public UniExprNode {
 public:
  FloorNode(DType dtype, Expr a) : UniExprNode(dtype, a) {
    ASSERT(dtype.is_int() || dtype.is_uint())
        << fmt::format("Floor output dtype should be Int or UInt, but get {}", std::string(dtype));
  }

  operator std::string() const override {
    return fmt::format("Floor({}, {})", std::string(this->dtype), std::string(this->a));
  }
};

using Floor = Ref<FloorNode>;

///=----------------------------------------------------------------------------=///
///
/// Ternary Operation IR Node
///
///=----------------------------------------------------------------------------=///

class SelectNode : public TerExprNode {
 public:
  SelectNode(Expr cond, Expr true_branch, Expr false_branch)
      : TerExprNode(true_branch->dtype, cond, true_branch, false_branch) {
    ASSERT(true_branch->dtype == false_branch->dtype)
        << fmt::format("Select expression expects the same dtype for both branches, but {} vs. {}",
                       std::string(true_branch->dtype), std::string(false_branch->dtype));
  }

  operator std::string() const override {
    return fmt::format("Select({}, {}, {}, {})", std::string(this->dtype), std::string(this->a),
                       std::string(this->b), std::string(this->c));
  }
};

using Select = Ref<SelectNode>;

///=----------------------------------------------------------------------------=///
///
/// Range IR Node
///
///=----------------------------------------------------------------------------=///

class RangeNode : public ExprNode {
 public:
  RangeNode(Expr beg, Expr extent, Expr step)
      : ExprNode(beg->dtype),
        beg(std::move(beg)),
        extent(std::move(extent)),
        step(std::move(step)) {
    ASSERT(this->beg.defined() && this->extent.defined() && this->step.defined());
    ASSERT(this->beg->dtype.is_int() || this->beg->dtype.is_uint());
    ASSERT(this->beg->dtype == this->extent->dtype && this->beg->dtype == this->step->dtype);
  }

  operator std::string() const override {
    return fmt::format("Range({}, {}, {})", std::string(this->beg), std::string(this->extent),
                       std::string(this->step));
  }

  Expr beg;
  Expr extent;
  Expr step;
};

using Range = Ref<RangeNode>;

class ExprListNode : public ExprNode {
 public:
  ExprListNode(std::vector<Expr> args)
      : ExprNode(DType::make("ignore")), value_list(std::move(args)) {
    for (auto arg : value_list) {
      ASSERT(arg.defined());
    }
  }

  ExprListNode(std::initializer_list<Expr> args)
      : ExprNode(DType::make("ignore")), value_list(std::move(args)) {
    for (auto arg : value_list) {
      ASSERT(arg.defined());
    }
  }

  void push_back(const Expr& e) {
    ASSERT(e.defined());
    this->value_list.push_back(e);
  }

  operator std::string() const override {
    std::vector<std::string> strs;
    for (auto v : this->value_list) {
      strs.push_back(std::string(v));
    }
    return fmt::format("ExprList({})", fmt::join(strs, ", "));
  }

  std::vector<Expr> value_list;
};

using ExprList = Ref<ExprListNode>;

///=----------------------------------------------------------------------------=///
///
/// Variable Operands Operation IR Node
///
///=----------------------------------------------------------------------------=///

// /// forward decl
// class ExprListNode;
// using ExprList = Ref<ExprListNode>;

class CondAllNode : public ExprNode {
 public:
  CondAllNode(ExprList ps) : ExprNode(DType::make("bool")), phases(std::move(ps)) {}
  CondAllNode(std::initializer_list<Expr> ps)
      : ExprNode(DType::make("bool")), phases(ExprList::make(ps)) {}
  CondAllNode(std::vector<Expr> ps) : ExprNode(DType::make("bool")), phases(ExprList::make(ps)) {}

  operator std::string() const override { return fmt::format("CondAll({})", std::string(phases)); }

  ExprList phases;
};

using CondAll = Ref<CondAllNode>;

class CondAnyNode : public ExprNode {
 public:
  CondAnyNode(ExprList ps) : ExprNode(DType::make("bool")), phases(std::move(ps)) {}
  CondAnyNode(std::initializer_list<Expr> ps)
      : ExprNode(DType::make("bool")), phases(ExprList::make(ps)) {}
  CondAnyNode(std::vector<Expr> ps) : ExprNode(DType::make("bool")), phases(ExprList::make(ps)) {}

  operator std::string() const override { return fmt::format("CondAny({})", std::string(phases)); }

  ExprList phases;
};

using CondAny = Ref<CondAnyNode>;

///=----------------------------------------------------------------------------=///
///
/// Constant IR Node
///
///=----------------------------------------------------------------------------=///

class ConstIntNode : public ConstExprNode {
 public:
  ConstIntNode(long long int value, int bits = 32, int lane = 1)
      : ConstExprNode(DType(DTypeKind::kInt, bits, lane)), value(value) {}

  operator std::string() const override {
    return fmt::format("Const({}, {})", this->value, std::string(this->dtype));
  }

  long long int value;
};

using ConstInt = Ref<ConstIntNode>;

class ConstUIntNode : public ConstExprNode {
 public:
  ConstUIntNode(unsigned long long int value, int bits = 32, int lane = 1)
      : ConstExprNode(DType(DTypeKind::kUInt, bits, lane)), value(value) {}

  operator std::string() const override {
    return fmt::format("Const({}, {})", this->value, std::string(this->dtype));
  }
  unsigned long long int value;
};

using ConstUInt = Ref<ConstUIntNode>;

class ConstFloatNode : public ConstExprNode {
 public:
  ConstFloatNode(double value, int bits = 32, int lane = 1)
      : ConstExprNode(DType(DTypeKind::kFloat, bits, lane)), value(value) {}

  operator std::string() const override {
    return fmt::format("Const({}, {})", this->value, std::string(this->dtype));
  }
  double value;
};

using ConstFloat = Ref<ConstFloatNode>;

class ConstBFloatNode : public ConstExprNode {
 public:
  ConstBFloatNode(double value, int bits = 16, int lane = 1)
      : ConstExprNode(DType(DTypeKind::kBFloat, bits, lane)), value(value) {}

  operator std::string() const override {
    return fmt::format("Const({}, {})", this->value, std::string(this->dtype));
  }
  double value;
};

using ConstBFloat = Ref<ConstBFloatNode>;

class ConstTFloatNode : public ConstExprNode {
 public:
  ConstTFloatNode(double value, int bits = 32, int lane = 1)
      : ConstExprNode(DType(DTypeKind::kTFloat, bits, lane)), value(value) {}

  operator std::string() const override {
    return fmt::format("Const({}, {})", this->value, std::string(this->dtype));
  }
  double value;
};

using ConstTFloat = Ref<ConstTFloatNode>;

class ConstStringNode : public ConstExprNode {
 public:
  ConstStringNode(std::string value) : ConstExprNode(DType::make("string")), value(value) {}
  operator std::string() const override {
    return fmt::format("Const({}, {})", this->value, std::string(this->dtype));
  }
  std::string value;
};

using ConstString = Ref<ConstStringNode>;

///=----------------------------------------------------------------------------=///
///
/// Variable IR Node
///
///=----------------------------------------------------------------------------=///

class VarNode : public MutableExprNode {
 public:
  VarNode(DType dtype, ConstString id) : MutableExprNode(std::move(dtype)), id(std::move(id)) {}
  VarNode(DType dtype, std::string id)
      : MutableExprNode(std::move(dtype)), id(ConstString::make(id)) {}

  operator std::string() const override {
    return fmt::format("Var({}, {})", std::string(this->id), std::string(this->dtype));
  }

  ConstString id;
};

using Var = Ref<VarNode>;

///=----------------------------------------------------------------------------=///
///
/// Iterator IR Node
///
///=----------------------------------------------------------------------------=///

enum class IterTypeKind : int { kSpatial = 0, kReduce = 1, kUnroll = 2, kZigzag = 3 };

std::string iter_type_to_string(IterTypeKind kind);

class IteratorNode : public ExprNode {
 public:
  IteratorNode(Var var, Range range, IterTypeKind iter_type)
      : ExprNode(var->dtype), var(var), range(range), iter_type(iter_type) {
    ASSERT(this->var.defined());
    ASSERT(this->range.defined());
  }

  operator std::string() const override {
    return fmt::format("Iter({}, {}, {})", std::string(this->var), std::string(this->range),
                       iter_type_to_string(this->iter_type));
  }

  Var var;
  Range range;
  IterTypeKind iter_type;
};

using Iterator = Ref<IteratorNode>;

///=----------------------------------------------------------------------------=///
///
/// Load IR Node
///
///=----------------------------------------------------------------------------=///

class NdLoadNode : public ExprNode {
 public:
  NdLoadNode(MemRef mem_ref, ExprList indices)
      : ExprNode(mem_ref->var->dtype), mem_ref(std::move(mem_ref)), indices(std::move(indices)) {
    ASSERT(this->mem_ref.defined());
    ASSERT(this->indices.defined());
  }

  operator std::string() const override {
    return fmt::format("NdLoad({}, {})", std::string(this->mem_ref), std::string(this->indices));
  }

  MemRef mem_ref;
  ExprList indices;
};

using NdLoad = Ref<NdLoadNode>;

class LoadNode : public ExprNode {
 public:
  LoadNode(MemRef mem_ref, Expr addr)
      : ExprNode(mem_ref->var->dtype), mem_ref(std::move(mem_ref)), addr(std::move(addr)) {
    ASSERT(this->mem_ref.defined());
    ASSERT(this->addr.defined());
  }

  operator std::string() const override {
    return fmt::format("Load({}, {})", std::string(this->mem_ref), std::string(this->addr));
  }

  MemRef mem_ref;
  Expr addr;
};

using Load = Ref<LoadNode>;

///=----------------------------------------------------------------------------=///
///
/// Map IR Node
///
///=----------------------------------------------------------------------------=///

class MapVarNode : public ExprNode {
 public:
  MapVarNode(Var var, Expr expr)
      : ExprNode(var->dtype), var(std::move(var)), expr(std::move(expr)) {
    ASSERT(this->var.defined());
    ASSERT(this->expr.defined());
  }

  operator std::string() const override {
    return fmt::format("MapVar({}, {})", std::string(this->var), std::string(this->expr));
  }

  Var var;
  Expr expr;
};

using MapVar = Ref<MapVarNode>;

///=----------------------------------------------------------------------------=///
///
/// Memory Slice IR Node
///
///=----------------------------------------------------------------------------=///

class SliceNode : public ExprNode {
 public:
  SliceNode(std::vector<Range> indices)
      : ExprNode(DType::make("ignore")), indices(std::move(indices)) {
    for (auto r : this->indices) {
      ASSERT(r.defined());
    }
  }

  operator std::string() const override {
    std::vector<std::string> strs;
    for (auto r : this->indices) {
      strs.push_back(std::string(r));
    }
    return fmt::format("Slice({})", fmt::join(strs, ", "));
  }

  std::vector<Range> indices;
};

using Slice = Ref<SliceNode>;

class MemSliceNode : public MemRefNode {
 public:
  MemSliceNode(Var var, Expr offset, Slice slice)
      : MemRefNode(std::move(var), std::move(offset)), slice(std::move(slice)) {
    ASSERT(this->slice.defined());
  }

  operator std::string() const override {
    return fmt::format("MemSlice({}, {}, {})", std::string(this->var), std::string(this->offset),
                       std::string(this->slice));
  }

  Slice slice;
};

using MemSlice = Ref<MemSliceNode>;

///=----------------------------------------------------------------------------=///
///
/// Call IR Node
///
///=----------------------------------------------------------------------------=///

class CallNode : public ExprNode {
 public:
  CallNode(DType dtype, ConstString func, ExprList args)
      : ExprNode(std::move(dtype)), func(std::move(func)), args(std::move(args)) {
    ASSERT(this->args.defined());
  }

  CallNode(DType dtype, std::string func, ExprList args)
      : ExprNode(std::move(dtype)), func(ConstString::make(func)), args(std::move(args)) {
    ASSERT(this->args.defined());
  }

  operator std::string() const override {
    return fmt::format("Call({}, {}, {})", std::string(this->dtype), std::string(this->func),
                       std::string(this->args));
  }

  ConstString func;
  ExprList args;
};

using Call = Ref<CallNode>;

///=----------------------------------------------------------------------------=///
///
/// Other IR Nodes
///
///=----------------------------------------------------------------------------=///

class PackValueNode : public ExprNode {
 public:
  PackValueNode(DType dtype, ExprList value_list)
      : ExprNode(dtype), value_list(std::move(value_list)) {}

  ExprList value_list;

  operator std::string() const override {
    return fmt::format("PackValue({}, {})", std::string(this->dtype),
                       std::string(this->value_list));
  }
};

using PackValue = Ref<PackValueNode>;

///=----------------------------------------------------------------------------=///
///
/// Convenient Helper Functions
///
///=----------------------------------------------------------------------------=///

Expr operator+(const Expr& a, const Expr& b);
Expr operator-(const Expr& a, const Expr& b);
Expr operator*(const Expr& a, const Expr& b);
Expr operator/(const Expr& a, const Expr& b);
Expr operator%(const Expr& a, const Expr& b);

Expr operator-(const Expr& a);

ConstInt const_int(long long int value, int bits = 32, int lanes = 1);
ConstUInt const_uint(unsigned long long int value, int bits = 32, int lanes = 1);
ConstFloat const_float(double value, int bits = 32, int lanes = 1);
ConstString const_string(std::string value);

Var var(const std::string dtype, const std::string& name = "");

}  // namespace domino

#endif
