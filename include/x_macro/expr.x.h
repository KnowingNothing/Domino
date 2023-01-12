#ifndef X_DECL_EXPR
#define X_DECL_EXPR(X) X_DECL_IR(X)
#endif

X_DECL_EXPR(Expr)
X_DECL_EXPR(BinExpr)
X_DECL_EXPR(UniExpr)
X_DECL_EXPR(TerExpr)
X_DECL_EXPR(ConstExpr)
X_DECL_EXPR(MutableExpr)
X_DECL_EXPR(MemRef)

#include <x_macro/bin_expr.x.h>

X_DECL_EXPR(Cast)
X_DECL_EXPR(Broadcast)
X_DECL_EXPR(Neg)
X_DECL_EXPR(Not)
X_DECL_EXPR(BitNot)
X_DECL_EXPR(Ceil)
X_DECL_EXPR(Floor)
X_DECL_EXPR(Select)
X_DECL_EXPR(CondAll)
X_DECL_EXPR(CondAny)
X_DECL_EXPR(ConstInt)
X_DECL_EXPR(ConstUInt)
X_DECL_EXPR(ConstFloat)
X_DECL_EXPR(ConstBFloat)
X_DECL_EXPR(ConstTFloat)
X_DECL_EXPR(ConstString)
X_DECL_EXPR(Range)
X_DECL_EXPR(ExprList)
X_DECL_EXPR(Var)
X_DECL_EXPR(Iterator)
X_DECL_EXPR(NdLoad)
X_DECL_EXPR(Load)
X_DECL_EXPR(MapVar)
X_DECL_EXPR(Slice)
X_DECL_EXPR(MemSlice)
X_DECL_EXPR(Call)
X_DECL_EXPR(PackValue)

#undef X_DECL_EXPR
