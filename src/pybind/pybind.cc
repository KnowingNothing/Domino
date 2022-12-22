#include <expr.h>
#include <fmt/core.h>
#include <ir_base.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ref.h>
#include <type_system/dtype.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, domino::Ref<T>);

namespace py = pybind11;

namespace domino {

PYBIND11_MODULE(dominoc, m) {
  /// bind type_system
  py::enum_<DTypeKind>(m, "DTypeKind")
      .value("Int", DTypeKind::kInt)
      .value("UInt", DTypeKind::kUInt)
      .value("Float", DTypeKind::kFloat)
      .value("BFloat", DTypeKind::kBFloat)
      .value("TFloat", DTypeKind::kTFloat)
      .value("MemRef", DTypeKind::kMemRef)
      .value("String", DTypeKind::kString)
      .value("Complex", DTypeKind::kComplex)
      .value("IGNORE", DTypeKind::kIGNORE)
      .value("UNKNOWN", DTypeKind::kUNKNOWN);
  py::class_<DType>(m, "DType")
      .def(py::init<DTypeKind, int, int>())
      .def("is_int", &DType::is_int)
      .def("is_uint", &DType::is_uint)
      .def("is_float", &DType::is_float)
      .def("is_bfloat", &DType::is_bfloat)
      .def("is_tfloat", &DType::is_tfloat)
      .def("is_memref", &DType::is_memref)
      .def("is_string", &DType::is_string)
      .def("is_complex", &DType::is_complex)
      .def("is_ignore", &DType::is_ignore)
      .def("is_unknown", &DType::is_unknown)
      .def("copy", &DType::copy)
      .def("with_lanes", &DType::with_lanes)
      .def("__repr__", [](const DType& d) { return std::string(d); })
      .def("__str__", [](const DType& d) { return std::string(d); })
      .def_readonly("type_kind", &DType::type_kind)
      .def_readonly("bits", &DType::bits)
      .def_readonly("lane", &DType::lane);

  /// submodule for IR
  py::module_ ir_m = m.def_submodule("ir", "IR Nodes of Domino");

  /// bind ir_base class
  py::class_<IRBaseNode, IRBase> pyIRBase(ir_m, "IRBase");
  pyIRBase.def(py::init<>())
      .def("__repr__", [](const IRBaseNode& d) { return std::string(d); })
      .def("__str__", [](const IRBaseNode& d) { return std::string(d); });

  /// bind expr classes
  py::class_<ExprNode, Expr> pyExpr(ir_m, "Expr", pyIRBase);
  pyExpr.def(py::init<DType>())
      .def("is_const", &ExprNode::IsConst)
      .def("__repr__", [](const ExprNode& d) { return std::string(d); })
      .def("__str__", [](const ExprNode& d) { return std::string(d); })
      .def_readonly("dtype", &ExprNode::dtype);

  /// bind binary operation
  py::class_<BinExprNode, BinExpr> pyBinExpr(ir_m, "BinExpr", pyExpr);
  pyBinExpr.def(py::init<DType, Expr, Expr>())
      .def("__repr__", [](const BinExprNode& d) { return std::string(d); })
      .def("__str__", [](const BinExprNode& d) { return std::string(d); })
      .def_readonly("a", &BinExprNode::a)
      .def_readonly("b", &BinExprNode::b);

  /// bind unary operation
  py::class_<UniExprNode, UniExpr> pyUniExpr(ir_m, "UniExpr", pyExpr);
  pyUniExpr.def(py::init<DType, Expr>())
      .def("__repr__", [](const UniExprNode& d) { return std::string(d); })
      .def("__str__", [](const UniExprNode& d) { return std::string(d); })
      .def_readonly("a", &UniExprNode::a);

  /// bind ternary operation
  py::class_<TerExprNode, TerExpr> pyTerExpr(ir_m, "TerExpr", pyExpr);
  pyTerExpr.def(py::init<DType, Expr, Expr, Expr>())
      .def("__repr__", [](const TerExprNode& d) { return std::string(d); })
      .def("__str__", [](const TerExprNode& d) { return std::string(d); })
      .def_readonly("a", &TerExprNode::a)
      .def_readonly("b", &TerExprNode::b)
      .def_readonly("c", &TerExprNode::c);

  /// bind constant expression
  py::class_<ConstExprNode, ConstExpr> pyConstExpr(ir_m, "ConstExpr", pyExpr);
  pyConstExpr.def(py::init<DType>())
      .def("__repr__", [](const ConstExprNode& d) { return std::string(d); })
      .def("__str__", [](const ConstExprNode& d) { return std::string(d); })
      .def("is_const", &ConstExprNode::IsConst);

  /// bind mutable expression
  py::class_<MutableExprNode, MutableExpr> pyMutableExpr(ir_m, "MutableExpr", pyExpr);
  pyMutableExpr.def(py::init<DType>())
      .def("__repr__", [](const MutableExprNode& d) { return std::string(d); })
      .def("__str__", [](const MutableExprNode& d) { return std::string(d); });

  /// bind variable
  py::class_<VarNode, Var> pyVar(ir_m, "Var", pyMutableExpr);

  /// bind memory reference
  py::class_<MemRefNode, MemRef> pyMemRef(ir_m, "MemRef", pyExpr);
  pyMemRef.def(py::init<Var, Expr>())
      .def("__repr__", [](const MemRefNode& d) { return std::string(d); })
      .def("__str__", [](const MemRefNode& d) { return std::string(d); })
      .def_readonly("var", &MemRefNode::var)
      .def_readonly("offset", &MemRefNode::offset);

  /// bind binary IR Node
#define BIND_BIN_IR_NODE(NAME)                                             \
  py::class_<NAME##Node, NAME>(ir_m, #NAME, pyBinExpr)                     \
      .def(py::init<Expr, Expr>())                                         \
      .def("__repr__", [](const NAME##Node& d) { return std::string(d); }) \
      .def("__str__", [](const NAME##Node& d) { return std::string(d); });

  BIND_BIN_IR_NODE(Add);
  BIND_BIN_IR_NODE(Sub);
  BIND_BIN_IR_NODE(Mul);
  BIND_BIN_IR_NODE(Div);
  BIND_BIN_IR_NODE(Mod);
  BIND_BIN_IR_NODE(FloorDiv);
  BIND_BIN_IR_NODE(FloorMod);
  BIND_BIN_IR_NODE(And);
  BIND_BIN_IR_NODE(Or);
  BIND_BIN_IR_NODE(XOr);
  BIND_BIN_IR_NODE(BitAnd);
  BIND_BIN_IR_NODE(BitOr);
  BIND_BIN_IR_NODE(BitXOr);
  BIND_BIN_IR_NODE(GT);
  BIND_BIN_IR_NODE(GE);
  BIND_BIN_IR_NODE(LT);
  BIND_BIN_IR_NODE(LE);
  BIND_BIN_IR_NODE(EQ);
  BIND_BIN_IR_NODE(NE);
#undef BIND_BIN_IR_NODE

  /// bind unary IR Node
#define BIND_UNI_IR_NODE(NAME)                                             \
  py::class_<NAME##Node, NAME>(ir_m, #NAME, pyUniExpr)                     \
      .def(py::init<Expr>())                                               \
      .def("__repr__", [](const NAME##Node& d) { return std::string(d); }) \
      .def("__str__", [](const NAME##Node& d) { return std::string(d); });

  BIND_UNI_IR_NODE(Neg);
  BIND_UNI_IR_NODE(Not);
  BIND_UNI_IR_NODE(BitNot);
#undef BIND_UNI_IR_NODE

  py::class_<CastNode, Cast>(ir_m, "Cast", pyUniExpr)
      .def(py::init<DType, Expr>())
      .def("__repr__", [](const CastNode& d) { return std::string(d); })
      .def("__str__", [](const CastNode& d) { return std::string(d); });

  py::class_<BroadcastNode, Broadcast>(ir_m, "Broadcast", pyUniExpr)
      .def(py::init<Expr, int>())
      .def("__repr__", [](const BroadcastNode& d) { return std::string(d); })
      .def("__str__", [](const BroadcastNode& d) { return std::string(d); });

  py::class_<CeilNode, Ceil>(ir_m, "Ceil", pyUniExpr)
      .def(py::init<DType, Expr>())
      .def("__repr__", [](const CeilNode& d) { return std::string(d); })
      .def("__str__", [](const CeilNode& d) { return std::string(d); });

  py::class_<FloorNode, Floor>(ir_m, "Floor", pyUniExpr)
      .def(py::init<DType, Expr>())
      .def("__repr__", [](const FloorNode& d) { return std::string(d); })
      .def("__str__", [](const FloorNode& d) { return std::string(d); });

  /// bind ternary IR Node
  py::class_<SelectNode, Select>(ir_m, "Select", pyTerExpr)
      .def(py::init<Expr, Expr, Expr>())
      .def("__repr__", [](const SelectNode& d) { return std::string(d); })
      .def("__str__", [](const SelectNode& d) { return std::string(d); });

  /// bind range
  py::class_<RangeNode, Range>(ir_m, "Range", pyExpr)
      .def(py::init<Expr, Expr, Expr>())
      .def("__repr__", [](const RangeNode& d) { return std::string(d); })
      .def("__str__", [](const RangeNode& d) { return std::string(d); })
      .def_readonly("beg", &RangeNode::beg)
      .def_readonly("extent", &RangeNode::extent)
      .def_readonly("step", &RangeNode::step);

  /// bind expression list
  py::class_<ExprListNode, ExprList>(ir_m, "ExprList", pyExpr)
      .def(py::init<std::vector<Expr>>())
      .def("__repr__", [](const ExprListNode& d) { return std::string(d); })
      .def("__str__", [](const ExprListNode& d) { return std::string(d); })
      .def("append", &ExprListNode::push_back)
      .def_readonly("value_list", &ExprListNode::value_list);

  /// bind variable operands IR Node
  py::class_<CondAllNode, CondAll>(ir_m, "CondAll", pyExpr)
      .def(py::init<std::vector<Expr>>())
      .def("__repr__", [](const CondAllNode& d) { return std::string(d); })
      .def("__str__", [](const CondAllNode& d) { return std::string(d); })
      .def_readonly("phases", &CondAllNode::phases);

  py::class_<CondAnyNode, CondAny>(ir_m, "CondAny", pyExpr)
      .def(py::init<std::vector<Expr>>())
      .def("__repr__", [](const CondAnyNode& d) { return std::string(d); })
      .def("__str__", [](const CondAnyNode& d) { return std::string(d); })
      .def_readonly("phases", &CondAnyNode::phases);

  /// bind const IR Node
  py::class_<ConstIntNode, ConstInt>(ir_m, "ConstInt", pyConstExpr)
      .def(py::init<long long int, int, int>())
      .def("__repr__", [](const ConstIntNode& d) { return std::string(d); })
      .def("__str__", [](const ConstIntNode& d) { return std::string(d); })
      .def_readonly("value", &ConstIntNode::value);

  py::class_<ConstUIntNode, ConstUInt>(ir_m, "ConstUInt", pyConstExpr)
      .def(py::init<unsigned long long int, int, int>())
      .def("__repr__", [](const ConstUIntNode& d) { return std::string(d); })
      .def("__str__", [](const ConstUIntNode& d) { return std::string(d); })
      .def_readonly("value", &ConstUIntNode::value);

  py::class_<ConstFloatNode, ConstFloat>(ir_m, "ConstFloat", pyConstExpr)
      .def(py::init<double, int, int>())
      .def("__repr__", [](const ConstFloatNode& d) { return std::string(d); })
      .def("__str__", [](const ConstFloatNode& d) { return std::string(d); })
      .def_readonly("value", &ConstFloatNode::value);

  py::class_<ConstBFloatNode, ConstBFloat>(ir_m, "ConstBFloat", pyConstExpr)
      .def(py::init<double, int, int>())
      .def("__repr__", [](const ConstBFloatNode& d) { return std::string(d); })
      .def("__str__", [](const ConstBFloatNode& d) { return std::string(d); })
      .def_readonly("value", &ConstBFloatNode::value);

  py::class_<ConstTFloatNode, ConstTFloat>(ir_m, "ConstTFloat", pyConstExpr)
      .def(py::init<double, int, int>())
      .def("__repr__", [](const ConstTFloatNode& d) { return std::string(d); })
      .def("__str__", [](const ConstTFloatNode& d) { return std::string(d); })
      .def_readonly("value", &ConstTFloatNode::value);

  py::class_<ConstStringNode, ConstString>(ir_m, "ConstString", pyConstExpr)
      .def(py::init<std::string>())
      .def("__repr__", [](const ConstStringNode& d) { return std::string(d); })
      .def("__str__", [](const ConstStringNode& d) { return std::string(d); })
      .def_readonly("value", &ConstStringNode::value);

  /// bind variable IR Node
  pyVar.def(py::init<DType, std::string>())
      .def("__repr__", [](const VarNode& d) { return std::string(d); })
      .def("__str__", [](const VarNode& d) { return std::string(d); })
      .def_readonly("id", &VarNode::id);

  /// bind iterator IR Node
  py::enum_<IterTypeKind>(ir_m, "IterTypeKind")
      .value("Spatial", IterTypeKind::kSpatial)
      .value("Reduce", IterTypeKind::kReduce);

  py::class_<IteratorNode, Iterator>(ir_m, "Iterator", pyExpr)
      .def(py::init<Var, Range, IterTypeKind>())
      .def("__repr__", [](const IteratorNode& d) { return std::string(d); })
      .def("__str__", [](const IteratorNode& d) { return std::string(d); })
      .def_readonly("var", &IteratorNode::var)
      .def_readonly("range", &IteratorNode::range)
      .def_readonly("iter_type", &IteratorNode::iter_type);

  /// bind load IR Node
  py::class_<NdLoadNode, NdLoad>(ir_m, "NdLoad", pyExpr)
      .def(py::init<MemRef, ExprList>())
      .def("__repr__", [](const NdLoadNode& d) { return std::string(d); })
      .def("__str__", [](const NdLoadNode& d) { return std::string(d); })
      .def_readonly("mem_ref", &NdLoadNode::mem_ref)
      .def_readonly("indices", &NdLoadNode::indices);

  py::class_<LoadNode, Load>(ir_m, "Load", pyExpr)
      .def(py::init<MemRef, Expr>())
      .def("__repr__", [](const LoadNode& d) { return std::string(d); })
      .def("__str__", [](const LoadNode& d) { return std::string(d); })
      .def_readonly("mem_ref", &LoadNode::mem_ref)
      .def_readonly("addr", &LoadNode::addr);

  /// bind map IR Node
  py::class_<MapVarNode, MapVar>(ir_m, "MapVar", pyExpr)
      .def(py::init<Var, Expr>())
      .def("__repr__", [](const MapVarNode& d) { return std::string(d); })
      .def("__str__", [](const MapVarNode& d) { return std::string(d); })
      .def_readonly("var", &MapVarNode::var)
      .def_readonly("expr", &MapVarNode::expr);

  /// bind memory slice IR Node
  py::class_<SliceNode, Slice>(ir_m, "Slice", pyExpr)
      .def(py::init<std::vector<Range>>())
      .def("__repr__", [](const SliceNode& d) { return std::string(d); })
      .def("__str__", [](const SliceNode& d) { return std::string(d); })
      .def_readonly("indices", &SliceNode::indices);

  py::class_<MemSliceNode, MemSlice>(ir_m, "MemSlice", pyMemRef)
      .def(py::init<Var, Expr, Slice>())
      .def("__repr__", [](const MemSliceNode& d) { return std::string(d); })
      .def("__str__", [](const MemSliceNode& d) { return std::string(d); })
      .def_readonly("slice", &MemSliceNode::slice);

  /// bind call IR Node
  py::class_<CallNode, Call>(ir_m, "Call", pyExpr)
      .def(py::init<DType, std::string, ExprList>())
      .def("__repr__", [](const CallNode& d) { return std::string(d); })
      .def("__str__", [](const CallNode& d) { return std::string(d); })
      .def_readonly("func", &CallNode::func)
      .def_readonly("args", &CallNode::args);
}

}  // namespace domino