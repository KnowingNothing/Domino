#include <expr.h>
#include <unordered_map>
#include <gtest/gtest.h>

using namespace domino;

TEST(test_ref, ref_equal) {
  Var v = var("int32", "a");
  Expr expr = v;
  Var as_var = expr.as<VarNode>();
  EXPECT_EQ(v, as_var);
  std::unordered_map<Var, Expr> m;
  m[v] = expr;
  EXPECT_EQ(m.at(v), expr);
}
