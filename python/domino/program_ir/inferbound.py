from dominoc import ir
from .scalar_expr import *

Stype = ir.Stype

__all__ = ["CoefNum", "SetGeneral", "SetConst", "SetVar"]


class CoefNum(ir.CoefNum):
    def __init__(self, value: int):
        ir.CoefNum.__init__(self, value)

    def __init__(self, a: ir.CoefNum):
        ir.CoefNum.__init__(self, a)

    def negate(self):
        ir.CoefNum.negate(self)


class SetGeneral(ir.SetGeneral):
    def __init__(self, type: Stype):
        ir.SetGeneral.__init__(self, type)


class SetConst(ir.SetConst, SetGeneral):
    def __init__(self):
        ir.SetConst.__init__(self)
        SetGeneral.__init__(self, Stype.SET_CONST)

    def __init__(self, a: int):
        ir.SetConst.__init__(self, a)
        SetGeneral.__init__(self, Stype.SET_CONST)

    def __init__(self, a: ConstInt):
        ir.SetConst.__init__(self, a)
        SetGeneral.__init__(self, Stype.SET_CONST)

    def __init__(self, a: Iterator):
        ir.SetConst.__init__(self, a)
        SetGeneral.__init__(self, Stype.SET_CONST)

    def __init__(self, a: ir.SetConst):
        ir.SetConst.__init__(self, a)
        SetGeneral.__init__(self, Stype.SET_CONST)

    def negate(self):
        ir.SetConst.negate(self)


class SetVar(ir.SetVar, SetGeneral):
    def __init__(self, a: Var):
        ir.SetVar.__init__(self, a)
        SetGeneral.__init__(self, Stype.SET_VAR)

    def negate(self):
        ir.SetVar.negate(self)
