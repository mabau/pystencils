from pystencils.backend.kernelcreation import KernelCreationContext
from pystencils.backend.ast.expressions import PsExpression, PsConstantExpr
from pystencils.backend.symbols import PsSymbol
from pystencils.backend.constants import PsConstant
from pystencils.backend.transformations import EliminateConstants

from pystencils.types.quick import Int, Fp

x, y, z = [PsExpression.make(PsSymbol(name)) for name in "xyz"]

f3p5 = PsExpression.make(PsConstant(3.5, Fp(32)))
f42 = PsExpression.make(PsConstant(42, Fp(32)))

f0 = PsExpression.make(PsConstant(0.0, Fp(32)))
f1 = PsExpression.make(PsConstant(1.0, Fp(32)))

i0 = PsExpression.make(PsConstant(0, Int(32)))
i1 = PsExpression.make(PsConstant(1, Int(32)))

i3 = PsExpression.make(PsConstant(3, Int(32)))
i12 = PsExpression.make(PsConstant(12, Int(32)))


def test_idempotence():
    ctx = KernelCreationContext()
    elim = EliminateConstants(ctx)

    expr = f42 * (f1 + f0) - f0
    result = elim(expr)
    assert isinstance(result, PsConstantExpr) and result.structurally_equal(f42)

    expr = (x + f0) * f3p5 + (f1 * y + f0) * f42
    result = elim(expr)
    assert result.structurally_equal(x * f3p5 + y * f42)

    expr = (f3p5 * f1) + (f42 * f1)
    result = elim(expr)
    #   do not fold floats by default
    assert expr.structurally_equal(f3p5 + f42)

    expr = f1 * x + f0 + (f0 + f0 + f1 + f0) * y
    result = elim(expr)
    assert result.structurally_equal(x + y)


def test_int_folding():
    ctx = KernelCreationContext()
    elim = EliminateConstants(ctx)

    expr = (i1 * x + i1 * i3) + i1 * i12
    result = elim(expr)
    assert result.structurally_equal((x + i3) + i12)

    expr = (i1 + i1 + i1 + i0 + i0 + i1) * (i1 + i1 + i1)
    result = elim(expr)
    assert result.structurally_equal(i12)


def test_zero_dominance():
    ctx = KernelCreationContext()
    elim = EliminateConstants(ctx)

    expr = (f0 * x) + (y * f0) + f1
    result = elim(expr)
    assert result.structurally_equal(f1)

    expr = (i3 + i12 * (x + y) + x / (i3 * y)) * i0
    result = elim(expr)
    assert result.structurally_equal(i0)
