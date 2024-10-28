from pystencils import Target

from pystencils.backend.ast.expressions import PsExpression
from pystencils.backend.ast.structural import PsAssignment, PsLoop, PsBlock
from pystencils.backend.kernelfunction import KernelFunction
from pystencils.backend.memory import PsSymbol, PsBuffer
from pystencils.backend.constants import PsConstant
from pystencils.backend.literals import PsLiteral
from pystencils.types.quick import Fp, SInt, UInt, Bool
from pystencils.backend.emission import CAstPrinter


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
        PsRem,
    )

    expr = PsBitwiseAnd(
        PsBitwiseXor(
            PsBitwiseXor(j, k),
            PsBitwiseOr(PsLeftShift(i, PsRightShift(j, k)), PsIntDiv(i, k)),
        )
        + PsRem(i, k),
        i,
    )
    code = cprint(expr)
    assert code == "(j ^ k ^ (i << (j >> k) | i / k)) + i % k & i"


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


def test_ternary():
    from pystencils.backend.ast.expressions import PsTernary
    from pystencils.backend.ast.expressions import PsNot, PsAnd, PsOr

    p, q = [PsExpression.make(PsSymbol(x, Bool())) for x in "pq"]
    x, y, z = [PsExpression.make(PsSymbol(x, Fp(32))) for x in "xyz"]
    cprint = CAstPrinter()

    expr = PsTernary(p, x, y)
    code = cprint(expr)
    assert code == "p ? x : y"

    expr = PsTernary(PsAnd(p, q), x + y, z)
    code = cprint(expr)
    assert code == "p && q ? x + y : z"

    expr = PsTernary(p, PsTernary(q, x, y), z)
    code = cprint(expr)
    assert code == "p ? (q ? x : y) : z"

    expr = PsTernary(p, x, PsTernary(q, y, z))
    code = cprint(expr)
    assert code == "p ? x : q ? y : z"

    expr = PsTernary(PsTernary(p, q, PsOr(p, q)), x, y)
    code = cprint(expr)
    assert code == "(p ? q : p || q) ? x : y"


def test_arrays():
    import sympy as sp
    from pystencils import Assignment
    from pystencils.backend.kernelcreation import KernelCreationContext, AstFactory

    ctx = KernelCreationContext(default_dtype=SInt(32))
    factory = AstFactory(ctx)
    cprint = CAstPrinter()

    arr_1d = factory.parse_sympy(Assignment(sp.Symbol("a1d"), sp.Tuple(1, 2, 3, 4, 5)))
    code = cprint(arr_1d)
    assert code == "int32_t a1d[5] = { 1, 2, 3, 4, 5 };"

    arr_2d = factory.parse_sympy(
        Assignment(sp.Symbol("a2d"), sp.Tuple((1, -1), (2, -2)))
    )
    code = cprint(arr_2d)
    assert code == "int32_t a2d[2][2] = { { 1, -1 }, { 2, -2 } };"

    arr_3d = factory.parse_sympy(
        Assignment(sp.Symbol("a3d"), sp.Tuple(((1, -1), (2, -2)), ((3, -3), (4, -4))))
    )
    code = cprint(arr_3d)
    assert (
        code
        == "int32_t a3d[2][2][2] = { { { 1, -1 }, { 2, -2 } }, { { 3, -3 }, { 4, -4 } } };"
    )
