#include <expr.h>
#include <gtest/gtest.h>

using namespace domino;

// TEST(test_ir_repr, simple_expr) {
//   auto a = Var::make("a");
//   auto b = Var::make("b");
//   auto t0 = UnOp::make("-", a);
//   auto t1 = BinOp::make("+", a, b);
//   auto t2 = BinOp::make("*", t0, t1);
//   auto t3 = BinOp::make("/", t2, IntConst::make(12));
//   auto t4 = BinOp::make("==", a, b);
//   auto t5 = IfExpr::make(t4, t1, t3);
//   EXPECT_EQ(repr(a), std::string("a"));
//   EXPECT_EQ(repr(b), std::string("b"));
//   EXPECT_EQ(repr(t0), std::string("(-a)"));
//   EXPECT_EQ(repr(t1), std::string("(a + b)"));
//   EXPECT_EQ(repr(t2), std::string("((-a) * (a + b))"));
//   EXPECT_EQ(repr(t3), std::string("(((-a) * (a + b)) / 12)"));
//   EXPECT_EQ(repr(t4), std::string("(a == b)"));
//   EXPECT_EQ(repr(t5), std::string("((a == b) ? (a + b) : (((-a) * (a + b)) / 12))"));
// }

// TEST(test_ir_repr, load_expr) {
//   auto load = Load::make("a", std::vector<Expr>({IntConst::make(1), Var::make("x")}));
//   EXPECT_EQ(repr(load), std::string("a[1, x]"));
// }

TEST(test_ir_repr, load_expr) {
  auto v = Var::make(DType::make("int32"), ConstString::make("a"));
  auto b = FloorDiv::make(v, v);
  EXPECT_EQ(repr(b), std::string("(a // a)"));
}

