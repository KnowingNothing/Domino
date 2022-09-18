#include <expr.h>
#include <gtest/gtest.h>
#include <printer.h>

using namespace domino;

TEST(test_print, complicated_expr) {
  auto a = Var::make("a");
  auto b = Var::make("b");
  auto t0 = UnOp::make("-", a);
  auto t1 = BinOp::make("+", a, b);
  auto t2 = BinOp::make("*", t0, t1);
  auto t3 = BinOp::make("/", t2, IntImm::make(12));
  auto t4 = BinOp::make("==", a, b);
  auto t5 = IfExpr::make(t4, t1, t3);
  EXPECT_EQ(to_string(a), std::string("a"));
  EXPECT_EQ(to_string(b), std::string("b"));
  EXPECT_EQ(to_string(t0), std::string("(-a)"));
  EXPECT_EQ(to_string(t1), std::string("(a + b)"));
  EXPECT_EQ(to_string(t2), std::string("((-a) * (a + b))"));
  EXPECT_EQ(to_string(t3), std::string("(((-a) * (a + b)) / 12)"));
  EXPECT_EQ(to_string(t4), std::string("(a == b)"));
  EXPECT_EQ(to_string(t5), std::string("((a == b) ? (a + b) : (((-a) * (a + b)) / 12))"));
}
