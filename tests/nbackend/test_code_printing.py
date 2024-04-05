from pystencils import Target

from pystencils.backend.ast.expressions import PsExpression, PsArrayAccess
from pystencils.backend.ast.structural import PsAssignment, PsLoop, PsBlock
from pystencils.backend.kernelfunction import KernelFunction
from pystencils.backend.symbols import PsSymbol
from pystencils.backend.constants import PsConstant
from pystencils.backend.arrays import PsLinearizedArray, PsArrayBasePointer
from pystencils.types.quick import Fp, SInt, UInt, Bool
from pystencils.backend.emission import CAstPrinter


# def test_basic_kernel():

#     u_arr = PsLinearizedArray("u", Fp(64), (..., ), (1, ))
#     u_size = PsExpression.make(u_arr.shape[0])
#     u_base = PsArrayBasePointer("u_data", u_arr)

#     loop_ctr = PsExpression.make(PsSymbol("ctr", UInt(32)))
#     one = PsExpression.make(PsConstant(1, SInt(32)))

#     update = PsAssignment(
#         PsArrayAccess(u_base, loop_ctr),
#         PsArrayAccess(u_base, loop_ctr + one) + PsArrayAccess(u_base, loop_ctr - one),
#     )

#     loop = PsLoop(
#         loop_ctr,
#         one,
#         u_size - one,
#         one,
#         PsBlock([update])
#     )

#     func = KernelFunction(PsBlock([loop]), Target.CPU, "kernel", set())

#     printer = CAstPrinter()
#     code = printer(func)

#     paramlist = func.get_parameters().params
#     params_str = ", ".join(f"{p.dtype} {p.name}" for p in paramlist)

#     assert code.find("(" + params_str + ")") >= 0
#     assert code.find("u_data[ctr] = u_data[ctr + 1] + u_data[ctr - 1];") >= 0


def test_arithmetic_precedence():
    (a, b, c, d, e, f) = [PsExpression.make(PsSymbol(x, Fp(64))) for x in "abcdef"]
    cprint = CAstPrinter()

    expr = (a + b) + (c + d)
    code = cprint(expr)
    assert code == "a + b + (c + d)"

    expr = ((a + b) + c) + d
    code = cprint(expr)
    assert code == "a + b + c + d"

    expr = a + (b + (c + d))
    code = cprint(expr)
    assert code == "a + (b + (c + d))"

    expr = a - (b - c) - d
    code = cprint(expr)
    assert code == "a - (b - c) - d"

    expr = a + b * (c + d * (e + f))
    code = cprint(expr)
    assert code == "a + b * (c + d * (e + f))"

    expr = (-a) + b + (-c) + -(e + f)
    code = cprint(expr)
    assert code == "-a + b + -c + -(e + f)"

    expr = (a / b) + (c / (d + e) * f)
    code = cprint(expr)
    assert code == "a / b + c / (d + e) * f"


def test_printing_integer_functions():
    (i, j, k) = [PsExpression.make(PsSymbol(x, UInt(64))) for x in "ijk"]
    cprint = CAstPrinter()

    from pystencils.backend.ast.expressions import (
        PsLeftShift,
        PsRightShift,
        PsBitwiseAnd,
        PsBitwiseOr,
        PsBitwiseXor,
        PsIntDiv,
    )

    expr = PsBitwiseAnd(
        PsBitwiseXor(
            PsBitwiseXor(j, k),
            PsBitwiseOr(PsLeftShift(i, PsRightShift(j, k)), PsIntDiv(i, k)),
        ),
        i,
    )
    code = cprint(expr)
    assert code == "(j ^ k ^ (i << (j >> k) | i / k)) & i"


def test_logical_precedence():
    from pystencils.backend.ast.expressions import PsNot, PsAnd, PsOr

    p, q, r = [PsExpression.make(PsSymbol(x, Bool())) for x in "pqr"]
    true = PsExpression.make(PsConstant(True, Bool()))
    false = PsExpression.make(PsConstant(False, Bool()))
    cprint = CAstPrinter()

    expr = PsNot(PsAnd(p, PsOr(q, r)))
    code = cprint(expr)
    assert code == "!(p && (q || r))"

    expr = PsAnd(PsAnd(p, q), PsAnd(q, r))
    code = cprint(expr)
    assert code == "p && q && (q && r)"

    expr = PsOr(PsAnd(true, p), PsOr(PsAnd(false, PsNot(q)), PsAnd(r, p)))
    code = cprint(expr)
    assert code == "true && p || (false && !q || r && p)"

    expr = PsAnd(PsOr(PsNot(p), PsNot(q)), PsNot(PsOr(true, false)))
    code = cprint(expr)
    assert code == "(!p || !q) && !(true || false)"


def test_relations_precedence():
    from pystencils.backend.ast.expressions import (
        PsNot,
        PsAnd,
        PsOr,
        PsEq,
        PsNe,
        PsLt,
        PsGt,
        PsLe,
        PsGe,
    )

    x, y, z = [PsExpression.make(PsSymbol(x, Fp(32))) for x in "xyz"]
    cprint = CAstPrinter()

    expr = PsAnd(PsEq(x, y), PsLe(y, z))
    code = cprint(expr)
    assert code == "x == y && y <= z"

    expr = PsOr(PsLt(x, y), PsLt(y, z))
    code = cprint(expr)
    assert code == "x < y || y < z"

    expr = PsAnd(PsNot(PsGe(x, y)), PsNot(PsLe(y, z)))
    code = cprint(expr)
    assert code == "!(x >= y) && !(y <= z)"

    expr = PsOr(PsNe(x, y), PsNot(PsGt(y, z)))
    code = cprint(expr)
    assert code == "x != y || !(y > z)"
