#ifndef DOMINO_RIGHT_THREAD_H
#define DOMINO_RIGHT_THREAD_H

#include <expr.h>

#include <algorithm>
#include <unordered_map>
#include <vector>
namespace domino {

enum Stype { SET_CONST, SET_VAR };

struct ElementFeature {
  int num;
  std::string min_str;
  std::string max_str;

  ElementFeature(int n, std::string str) : num(n), min_str(str), max_str(str) {}

  ElementFeature(const ElementFeature& e) : num(e.num), min_str(e.min_str), max_str(e.max_str) {}
};

template <typename T1, typename T2>
class Term {
 public:
  T1 coef;
  std::vector<T2> element;
  ElementFeature feat;

  Term(T1 c, T2 element_one, std::string id) : coef(c), feat(1, id) {
    element.push_back(element_one);
  }

  Term(const Term<T1, T2>& t) : coef(T1::make(t.coef)), feat(t.feat) {
    for (int i = 0; i < t.element.size(); ++i) element.push_back(t.element[i]);
  }

  void mul(Term<T1, T2> a) {
    coef->Mul(a.coef);
    ASSERT(a.feat.num == a.element.size());
    feat.num += a.feat.num;
    for (int i = 0; i < a.element.size(); ++i) element.push_back(a.element[i]);
    feat.min_str = min(feat.min_str, a.feat.min_str);
    feat.max_str = max(feat.max_str, a.feat.max_str);
  }

  bool ifEqual(std::vector<T2>* v) {
    std::unordered_map<std::string, int> umap;
    for (int i = 0; i < element.size(); ++i) {
      std::string key = element[i]->getID();
      auto it = umap.find(key);
      if (it == umap.end())
        umap.emplace(key, 1);
      else
        it->second++;
    }
    for (int i = 0; i < v->size(); ++i) {
      std::string id = v->at(i)->getID();
      auto it = umap.find(id);
      if (it == umap.end()) {
        umap.clear();
        return false;
      }
      it->second--;
      if (it->second == 0) umap.erase(it);
    }
    ASSERT(umap.empty());
    return true;
  }

  bool operator<(const Term<T1, T2>& obj) { return feat.num < obj.feat.num; }

  int elementNum() {
    if (coef->ifOne()) return element.size();
    return 1 + element.size();
  }

  operator std::string() const {
    std::vector<std::string> strs;
    strs.push_back(std::string(coef));
    for (auto v : this->element) {
      strs.push_back(v->getID());
    }
    strs.push_back(std::to_string(feat.num));
    strs.push_back(feat.max_str);
    strs.push_back(feat.min_str);
    return fmt::format("Term({})", fmt::join(strs, ", "));
  }
};

/*
<CoefNum, Iterator> <SetConst, Var>
*/
template <typename T1, typename T2>
class TermSet {
 public:
  std::vector<Term<T1, T2>> terms;

  TermSet() {}

  TermSet(T1 c, T2 element_one, std::string id) { terms.push_back(Term(c, element_one, id)); }

  // 直接assign或=会导致关联的是对方的term的coef，后续产生不可想象的错误
  void safeAssign(std::vector<Term<T1, T2>> t) {
    for (int i = 0; i < t.size(); ++i) {
      // 已经改了Term的复制构造函数，会新生成Coef
      terms.push_back(t[i]);
    }
  }

  void MulJunior(T1 coef) {
    for (int i = 0; i < terms.size(); ++i) terms[i].coef->Mul(coef);
    this->eliminateZero();
  }

  void Mul(TermSet<T1, T2> a, T1 a_coef, T1 my_coef) {
    int a_num = a.terms.size(), my_num = terms.size();
    // 边界情况
    if (a_num == 0) {
      if (my_num == 0) return;
      this->MulJunior(a_coef);
      return;
    }
    if (my_num == 0) {
      a.MulJunior(my_coef);
      this->safeAssign(a.terms);
      return;
    }

    // 正常
    // terms*a
    TermSet<T1, T2> res, tmp;
    for (int i = 0; i < a_num; ++i) {
      tmp.terms.clear();
      tmp.safeAssign(terms);
      for (int j = 0; j < my_num; ++j)
        tmp.terms[j].mul(a.terms[i]);  // 正常情况下是不用考虑系数有0的情况的
      res.Merge(tmp);                  // Merge里消0了
    }

    // terms*a_coef
    this->MulJunior(a_coef);  // terms变了

    // a*my_coef
    a.MulJunior(my_coef);

    // 合并
    this->Merge(res);
    this->Merge(a);
  }

  void Merge(TermSet<T1, T2> a) {
    int a_num = a.terms.size(), my_num = terms.size();
    // 边界情况
    if (a_num == 0) return;
    if (my_num == 0) {
      this->safeAssign(a.terms);
      return;
    }

    // 都有元素
    for (int i = 0; i < a_num; ++i) {
      Term<T1, T2>* p = &(a.terms[i]);
      bool flag = false;
      for (int j = 0; j < my_num; ++j) {
        ElementFeature* my = &(terms[j].feat);
        if (my->num < p->feat.num) continue;
        if (my->num > p->feat.num) break;
        if (my->min_str != p->feat.min_str || my->max_str != p->feat.max_str) continue;
        if (terms[j].ifEqual(&(p->element))) {
          flag = true;
          terms[j].coef->Merge(p->coef);
          break;
        }
      }
      if (!flag) terms.push_back(*p);
    }
    // 排序
    std::sort(terms.begin(), terms.end());
    // 去0
    this->eliminateZero();
  }

  void eliminateZero() {
    typename std::vector<Term<T1, T2>>::iterator it = terms.begin();
    while (it != terms.end()) {
      if (it->coef->ifZero())
        it = terms.erase(it);
      else
        it += 1;
    }
  }

  void negate() {
    for (int i = 0; i < terms.size(); ++i) terms[i].coef->negate();
  }

  bool ifZero() { return terms.size() == 0; }

  int elementNum() {
    int res = 0;
    for (int i = 0; i < terms.size(); ++i) res += terms[i].elementNum();
    return res;
  }

  operator std::string() const {
    std::vector<std::string> strs;
    for (auto v : this->terms) {
      strs.push_back(std::string(v));
    }
    return fmt::format("TermSet({})", fmt::join(strs, ", "));
  }
};

class CoefNumNode : public IRBaseNode {
 public:
  long long int value;

  CoefNumNode(long long int t) : value(t) {}

  CoefNumNode(Ref<CoefNumNode> a) : value(a->value) {}

  void Mul(Ref<CoefNumNode> a) { value *= a->value; }

  void Merge(Ref<CoefNumNode> a) { value += a->value; }

  void negate() { value *= -1; }

  bool ifZero() { return value == 0; }

  bool ifOne() { return (value == 1 || value == -1); }

  int elementNum() {
    if (value == 0) return 0;
    return 1;
  }

  operator std::string() const override { return fmt::format("CoefNum({})", value); }
};
using CoefNum = Ref<CoefNumNode>;

class SetGeneralNode : public IRBaseNode {
 public:
  Stype stype;

  SetGeneralNode(Stype s) : stype(s) {}

  operator std::string() const override { return fmt::format("SetGeneral()"); }
};
using SetGeneral = Ref<SetGeneralNode>;

class SetConstNode : public SetGeneralNode {
 public:
  CoefNum cons_int;
  TermSet<CoefNum, Iterator> ts;

  SetConstNode() : SetGeneralNode(SET_CONST) {}

  SetConstNode(long long int a) : SetGeneralNode(SET_CONST), cons_int(CoefNum::make(a)) {}

  SetConstNode(ConstInt e) : SetGeneralNode(SET_CONST), cons_int(CoefNum::make(int(e->value))) {}

  SetConstNode(Iterator e)
      : SetGeneralNode(SET_CONST),
        cons_int(CoefNum::make(0)),
        ts(TermSet<CoefNum, Iterator>(CoefNum::make(1), e, std::string(e->var->id))) {}

  // 没用到：为了写测试更方便
  SetConstNode(Ref<SetConstNode> a)
      : SetGeneralNode(a->stype),
        cons_int(CoefNum::make(a->cons_int)),
        ts(TermSet<CoefNum, Iterator>(a->ts)) {}

  void Mul(Ref<SetConstNode> a) {
    ts.Mul(a->ts, a->cons_int, cons_int);
    cons_int->Mul(a->cons_int);
  }

  void Merge(Ref<SetConstNode> a) {
    ts.Merge(a->ts);
    cons_int->Merge(a->cons_int);
  }

  void negate() {
    ts.negate();
    cons_int->negate();
  }

  bool ifZero() { return cons_int->ifZero() && ts.ifZero(); }

  bool ifNumber() { return ts.ifZero(); }

  bool ifOne() { return (cons_int->ifOne() && ts.ifZero()); }

  int elementNum() { return cons_int->elementNum() + ts.elementNum(); }

  operator std::string() const override {
    return fmt::format("SetConst({}, {})", std::string(cons_int), std::string(ts));
  }
};
using SetConst = Ref<SetConstNode>;

class SetVarNode : public SetGeneralNode {
 public:
  TermSet<SetConst, Var> ts;
  SetConst cons_set;

  SetVarNode(Var e)
      : SetGeneralNode(SET_VAR),
        cons_set(SetConst::make(0)),
        ts(TermSet<SetConst, Var>(SetConst::make(1), e, std::string(e->id))) {}

  void Mul(Ref<SetVarNode> a) {
    ts.Mul(a->ts, a->cons_set, cons_set);
    cons_set->Mul(a->cons_set);
  }

  void Mul(SetConst a) {
    ts.MulJunior(a);
    cons_set->Mul(a);
  }

  void Merge(Ref<SetVarNode> a) {
    ts.Merge(a->ts);
    cons_set->Merge(a->cons_set);
  }

  void negate() {
    ts.negate();
    cons_set->negate();
  }

  bool ifConst() { return ts.ifZero(); }

  int elementNum() { return cons_set->elementNum() + ts.elementNum(); }

  operator std::string() const override {
    return fmt::format("SetVar({}, {})", std::string(cons_set), std::string(ts));
  }
};
using SetVar = Ref<SetVarNode>;

}  // namespace domino
#endif