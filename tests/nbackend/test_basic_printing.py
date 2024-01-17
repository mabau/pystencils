import pytest

from pystencils import Target

from pystencils.nbackend.ast import *
from pystencils.nbackend.typed_expressions import *
from pystencils.nbackend.arrays import PsLinearizedArray, PsArrayBasePointer, PsArrayAccess
from pystencils.nbackend.types.quick import *
from pystencils.nbackend.emission import CPrinter

def test_basic_kernel():

    u_arr = PsLinearizedArray("u", Fp(64), (..., ), (1, ))
    u_size = u_arr.shape[0]
    u_base = PsArrayBasePointer("u_data", u_arr)

    loop_ctr = PsTypedVariable("ctr", UInt(32))
    one = PsTypedConstant(1, SInt(32))

    update = PsAssignment(
        PsLvalueExpr(PsArrayAccess(u_base, loop_ctr)),
        PsExpression(PsArrayAccess(u_base, loop_ctr + one) + PsArrayAccess(u_base, loop_ctr - one)),
    )

    loop = PsLoop(
        PsSymbolExpr(loop_ctr),
        PsExpression(one),
        PsExpression(u_size - one),
        PsExpression(one),
        PsBlock([update])
    )

    func = PsKernelFunction(PsBlock([loop]), target=Target.CPU)

    printer = CPrinter()
    code = printer.print(func)

    paramlist = func.get_parameters().params
    params_str = ", ".join(f"{p.dtype} {p.name}" for p in paramlist)

    assert code.find("(" + params_str + ")") >= 0
    
    assert code.find("u_data[ctr] = u_data[ctr + 1] + u_data[ctr + -1];") >= 0

