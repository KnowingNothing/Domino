#include <analysis/fusion.h>
#include <arch.h>
#include <block.h>
#include <codegen/codegen.h>
#include <expr.h>
#include <fmt/core.h>
#include <ir_base.h>
#include <kernel.h>
#include <pass/flatten.h>
#include <pass/prod_consum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ref.h>
#include <simplify.h>
#include <stmt.h>
#include <type_system/dtype.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, domino::Ref<T>);

namespace py = pybind11;

namespace domino {

using namespace codegen;

void bindTypeSystem(py::module_& m) {
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
}

void bindIR(py::module_& m) {
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
      .def("__mul__", [](const Expr& a, const Expr& b) { return Mul::make(a, b); })
      .def("__mul__", [](const int a, const Expr& b) { return Mul::make(ConstInt::make(a), b); })
      .def("__floordiv__", [](const Expr& a, const Expr& b) { return FloorDiv::make(a, b); })
      .def("__floordiv__",
           [](const Expr& a, const int b) { return FloorDiv::make(a, ConstInt::make(b)); })
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

  /// bind value reference
  py::class_<ValueRefNode, ValueRef> pyValueRef(ir_m, "ValueRef", pyExpr);
  pyValueRef.def(py::init<Var>())
      .def("__repr__", [](const ValueRefNode& d) { return std::string(d); })
      .def("__str__", [](const ValueRefNode& d) { return std::string(d); })
      .def_readonly("var", &ValueRefNode::var);

  /// bind array reference
  py::class_<ArrayRefNode, ArrayRef> pyArrayRef(ir_m, "ArrayRef", pyExpr);
  pyArrayRef.def(py::init<Var, ExprList>())
      .def("__repr__", [](const ArrayRefNode& d) { return std::string(d); })
      .def("__str__", [](const ArrayRefNode& d) { return std::string(d); })
      .def_readonly("var", &ArrayRefNode::var)
      .def_readonly("args", &ArrayRefNode::args);

  /// bind binary IR Node
#define X_DECL_BIN_EXPR(NAME)                                              \
  py::class_<NAME##Node, NAME>(ir_m, #NAME, pyBinExpr)                     \
      .def(py::init<Expr, Expr>())                                         \
      .def("__repr__", [](const NAME##Node& d) { return std::string(d); }) \
      .def("__str__", [](const NAME##Node& d) { return std::string(d); });
#include <x_macro/bin_expr.x.h>

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

  py::class_<ConstVarNode, ConstVar>(ir_m, "ConstVar", pyVar)
      .def(py::init<DType, std::string>())
      .def("__repr__", [](const ConstVarNode& d) { return std::string(d); })
      .def("__str__", [](const ConstVarNode& d) { return std::string(d); })
      .def_readonly("id", &ConstVarNode::id);

  /// bind iterator IR Node
  py::enum_<IterTypeKind>(ir_m, "IterTypeKind")
      .value("Spatial", IterTypeKind::kSpatial)
      .value("Reduce", IterTypeKind::kReduce)
      .value("Unroll", IterTypeKind::kUnroll)
      .value("Zigzag", IterTypeKind::kZigzag)
      .value("Tensorized", IterTypeKind::kTensorized);

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

  /// bind pack_value IR Node
  py::class_<PackValueNode, PackValue>(ir_m, "PackValue", pyExpr)
      .def(py::init<DType, ExprList>())
      .def("__repr__", [](const PackValueNode& d) { return std::string(d); })
      .def("__str__", [](const PackValueNode& d) { return std::string(d); })
      .def_readonly("value_list", &PackValueNode::value_list);

  /// bind stmt classes
  py::class_<StmtNode, Stmt> pyStmt(ir_m, "Stmt", pyIRBase);
  pyStmt.def(py::init<>())
      .def("__repr__", [](const StmtNode& d) { return std::string(d); })
      .def("__str__", [](const StmtNode& d) { return std::string(d); });

  /// bind NdStore IR Node
  py::class_<NdStoreNode, NdStore>(ir_m, "NdStore", pyStmt)
      .def(py::init<MemRef, ExprList, Expr>())
      .def("__repr__", [](const NdStoreNode& d) { return std::string(d); })
      .def("__str__", [](const NdStoreNode& d) { return std::string(d); })
      .def_readonly("mem_ref", &NdStoreNode::mem_ref)
      .def_readonly("indices", &NdStoreNode::indices)
      .def_readonly("value", &NdStoreNode::value);

  /// bind Store IR Node
  py::class_<StoreNode, Store>(ir_m, "Store", pyStmt)
      .def(py::init<MemRef, Expr, Expr>())
      .def("__repr__", [](const StoreNode& d) { return std::string(d); })
      .def("__str__", [](const StoreNode& d) { return std::string(d); })
      .def_readonly("mem_ref", &StoreNode::mem_ref)
      .def_readonly("addr", &StoreNode::addr)
      .def_readonly("value", &StoreNode::value);

  /// bind Evaluate IR Node
  py::class_<EvaluateNode, Evaluate>(ir_m, "Evaluate", pyStmt)
      .def(py::init<Expr>())
      .def("__repr__", [](const EvaluateNode& d) { return std::string(d); })
      .def("__str__", [](const EvaluateNode& d) { return std::string(d); })
      .def_readonly("expr", &EvaluateNode::expr);

  /// bind block classes
  py::class_<BlockNode, Block> pyBlock(ir_m, "Block", pyIRBase);
  pyBlock.def(py::init<>());

  /// bind AttrBlock IR Node
  py::class_<AttrBlockNode, AttrBlock>(ir_m, "AttrBlock", pyBlock)
      .def(py::init<std::string, Var, Expr, Block>())
      .def_readonly("key", &AttrBlockNode::key)
      .def_readonly("obj", &AttrBlockNode::obj)
      .def_readonly("value", &AttrBlockNode::value)
      .def_readonly("body", &AttrBlockNode::body);

  /// bind NdForBlock IR Node
  py::class_<NdForBlockNode, NdForBlock>(ir_m, "NdForBlock", pyBlock)
      .def(py::init<std::vector<Iterator>, Block, std::string>())
      .def_readonly("iters", &NdForBlockNode::iters)
      .def_readonly("body", &NdForBlockNode::body)
      .def_readonly("compute_level", &NdForBlockNode::compute_level);

  /// bind ForBlock IR Node
  py::class_<ForBlockNode, ForBlock>(ir_m, "ForBlock", pyBlock)
      .def(py::init<Iterator, Block, std::string>())
      .def_readonly("iter", &ForBlockNode::iter)
      .def_readonly("body", &ForBlockNode::body)
      .def_readonly("compute_level", &ForBlockNode::compute_level);

  /// bind BranchBlock IR Node
  py::class_<BranchBlockNode, BranchBlock>(ir_m, "BranchBlock", pyBlock)
      .def(py::init<Expr, Block, Block>())
      .def_readonly("cond", &BranchBlockNode::cond)
      .def_readonly("true_branch", &BranchBlockNode::true_branch)
      .def_readonly("false_branch", &BranchBlockNode::false_branch);

  /// bind SeqBlock IR Node
  py::class_<SeqBlockNode, SeqBlock>(ir_m, "SeqBlock", pyBlock)
      .def(py::init<Block, Block>())
      .def_readonly("first", &SeqBlockNode::first)
      .def_readonly("second", &SeqBlockNode::second);

  /// bind SpatialBlock IR Node
  py::class_<SpatialBlockNode, SpatialBlock>(ir_m, "SpatialBlock", pyBlock)
      .def(py::init<std::vector<Block>, std::vector<ConstString>>())
      .def_readonly("blocks", &SpatialBlockNode::blocks)
      .def_readonly("spatial_bindings", &SpatialBlockNode::spatial_bindings);

  /// bind AtomBlock IR Node
  py::class_<AtomBlockNode, AtomBlock>(ir_m, "AtomBlock", pyBlock)
      .def(py::init<Stmt>())
      .def_static("make_null_block", &AtomBlockNode::makeNullBlock)
      .def("is_null_block", &AtomBlockNode::isNullBlock)
      .def("get_stmt", &AtomBlockNode::getStmt);

  /// bind ReMapBlock IR Node
  py::class_<ReMapBlockNode, ReMapBlock>(ir_m, "ReMapBlock", pyBlock)
      .def(py::init<std::vector<MapVar>, Block>())
      .def_readonly("mappings", &ReMapBlockNode::mappings)
      .def_readonly("body", &ReMapBlockNode::body);

  /// bind NdAllocBlock IR Node
  py::class_<NdAllocBlockNode, NdAllocBlock>(ir_m, "NdAllocBlock", pyBlock)
      .def(py::init<Var, std::vector<Expr>, ConstString, Block>())
      .def_readonly("var", &NdAllocBlockNode::var)
      .def_readonly("shape", &NdAllocBlockNode::shape)
      .def_readonly("memory_scope", &NdAllocBlockNode::memory_scope)
      .def_readonly("body", &NdAllocBlockNode::body);

  /// bind AllocBlock IR Node
  py::class_<AllocBlockNode, AllocBlock>(ir_m, "AllocBlock", pyBlock)
      .def(py::init<Var, Expr, ConstString, Block>())
      .def_readonly("var", &AllocBlockNode::var)
      .def_readonly("length", &AllocBlockNode::length)
      .def_readonly("memory_scope", &AllocBlockNode::memory_scope)
      .def_readonly("body", &AllocBlockNode::body);

  /// bind kernel classes
  py::class_<KernelSignatureNode, KernelSignature>(ir_m, "KernelSignature", pyIRBase)
      .def(py::init<std::string, std::vector<Var>, std::vector<Var>>())
      .def_readonly("kernel_name", &KernelSignatureNode::kernel_name)
      .def_readonly("kernel_args", &KernelSignatureNode::kernel_args);

  py::class_<KernelNode, Kernel> pyKernel(ir_m, "Kernel", pyIRBase);
  pyKernel.def(py::init<KernelSignature, Block>())
      .def_readonly("signature", &KernelNode::signature)
      .def_readonly("body", &KernelNode::body)
      .def_readwrite("source", &KernelNode::source)
      .def("compiled", &KernelNode::compiled)
      .def("gen_function", &KernelNode::genFunction)
      .def("gen_signature", &KernelNode::genSignature);

  /// bind architecture classes
  py::class_<ArchNode, Arch> pyArch(ir_m, "Arch", pyIRBase);
  pyArch.def(py::init<>());

  py::class_<MemoryLevelNode, MemoryLevel>(ir_m, "MemoryLevel", pyArch)
      .def(py::init<ConstInt, Block, std::vector<Arch>>())
      .def_readonly("memory_level", &MemoryLevelNode::memory_level)
      .def_readwrite("block", &MemoryLevelNode::block)
      .def_readwrite("sub_levels", &MemoryLevelNode::sub_levels)
      .def_readwrite("scope", &MemoryLevelNode::scope)
      .def_readwrite("annotation", &MemoryLevelNode::annotation);

  py::class_<ComputeLevelNode, ComputeLevel>(ir_m, "ComputeLevel", pyArch)
      .def(py::init<ConstInt, Block, std::vector<Arch>>())
      .def_readonly("compute_level", &ComputeLevelNode::compute_level)
      .def_readwrite("block", &ComputeLevelNode::block)
      .def_readwrite("sub_levels", &ComputeLevelNode::sub_levels)
      .def_readwrite("produce_var", &ComputeLevelNode::produce_var);

  /// bind IRPrinter function
  ir_m.def("print_ir", &repr, "Function that prints the IR.");

  /// bind ExprSimplifyPattern
  py::class_<ExprSimplifyPattern>(ir_m, "ExprSimplifyPattern")
      .def(py::init<Expr, Expr>())
      .def_readonly("old", &ExprSimplifyPattern::old)
      .def_readonly("replace", &ExprSimplifyPattern::replace);

  /// bind ExprSimplifyMatchPattern function
  ir_m.def("expr_simplify_match_pattern", &ExprSimplifyMatchPattern,
           "Function that performs pattern matching for simplify.");

  /// bind GetExprSimplifyMatchPatterns function
  ir_m.def("get_expr_simplify_match_patterns", &GetExprSimplifyMatchPatterns,
           "Function that performs pattern matching for simplify and returns the mapping.");

  /// bind SubstituteExpr function
  ir_m.def("substitute_expr", &SubstituteExpr,
           "Function that substitutes expression according to mapping.");

  /// bind SubstituteStmt function
  ir_m.def("substitute_stmt", &SubstituteStmt,
           "Function that substitutes statement according to mapping.");

  /// bind SubstituteBlock function
  ir_m.def("substitute_block", &SubstituteBlock,
           "Function that substitutes block according to mapping.");

  /// bind SubstituteIR function
  ir_m.def("substitute_ir", &SubstituteIR, "Function that substitutes IR according to mapping.");

  /// bind ExprSimplifier class
  //   py::class_<ExprSimplifier>(ir_m, "ExprSimplifier")
  //     // .def(py::init<>())
  //     .def_static("patterns_", &ExprSimplifier::patterns_,
  //     py::return_value_policy::reference_internal);

  /// bind SimplifyExpr function
  ir_m.def("simplify_expr", &SimplifyExpr,
           "Function that simplifies expressions according to a list of inner rules.");

  /// bind Simplify function
  ir_m.def("simplify", &Simplify,
           "Function that simplifies IR according to a list of inner rules.");

  /// bind replicate
  ir_m.def("replicate", &replicate, "Replicate an IR exactly.");
}

void bindCodeGen(py::module_& m) {
  /// submodule for CodeGen
  py::module_ gen_m = m.def_submodule("codegen", "Codegen of Domino");

  /// bind codegen_c
  gen_m.def("codegen_c", &codegen_c, "Codegen function for C source code.");

  /// bind codegen_arm_m
  gen_m.def("codegen_arm_m", &codegen_arm_m, "Codegen function for ARM Cortex M processor.");

  /// bind codegen_tileflow
  gen_m.def("codegen_tileflow", &codegen_tileflow, "Codegen function for TileFlow.");
}

void bindPass(py::module_& m) {
  /// submodule for pass
  py::module_ pass_m = m.def_submodule("passes", "Pass in Domino");

  /// bind flatten_array_access
  pass_m.def("flatten_array_access", &pass::FlattenArrayAccess, "Flatten array access.");

  /// bind get_input_tensor_vars
  pass_m.def("get_input_tensor_vars", &pass::GetInputTensorVars, "Get the input tensor vars.");

  /// bind get_input_tensor_indices
  pass_m.def("get_input_tensor_indices", &pass::GetInputTensorIndices,
             "Get the indics of an input tensor.");
}

void bindAnalysis(py::module_& m) {
  py::module_ ana_m = m.def_submodule("analysis", "Analysis in Domino");

  py::class_<analysis::MemoryLevelTreeNode, analysis::MemoryLevelTree> pyMemLevelTree(
      ana_m, "MemoryLevelTree");
  pyMemLevelTree.def(py::init<std::vector<int>, Var, std::unordered_map<Var, Range>>())
      .def("cut", &analysis::MemoryLevelTreeNode::Cut)
      .def("merge", &analysis::MemoryLevelTreeNode::Merge)
      .def("get_available_levels", &analysis::MemoryLevelTreeNode::GetAvailableLevels)
      .def("memory_tiling", &analysis::MemoryLevelTreeNode::MemoryTiling)
      .def("least_common_ancestor", &analysis::MemoryLevelTreeNode::LeastCommonAncestor)
      .def("set_bounds", &analysis::MemoryLevelTreeNode::SetBounds)
      .def_readonly("root", &analysis::MemoryLevelTreeNode::root)
      .def_readonly("merged", &analysis::MemoryLevelTreeNode::merged)
      .def_readonly("initial_levels", &analysis::MemoryLevelTreeNode::initial_levels)
      .def_readonly("tensor_var", &analysis::MemoryLevelTreeNode::tensor_var)
      .def_readonly("var_map", &analysis::MemoryLevelTreeNode::var_map)
      .def_readonly("bounds", &analysis::MemoryLevelTreeNode::bounds);

  ana_m.def("generate_merged_memory_level_trees", &analysis::generateMergedMemoryLevelTrees,
            "Generate possible merged memory level trees.");
}

PYBIND11_MODULE(dominoc, m) {
  bindTypeSystem(m);

  bindIR(m);

  bindCodeGen(m);

  bindPass(m);

  bindAnalysis(m);
}

}  // namespace domino