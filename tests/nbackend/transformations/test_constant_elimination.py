from pystencils.backend.kernelcreation import KernelCreationContext, Typifier
from pystencils.backend.ast.expressions import PsExpression, PsConstantExpr
from pystencils.backend.symbols import PsSymbol
from pystencils.backend.constants import PsConstant
from pystencils.backend.transformations import EliminateConstants

from pystencils.backend.ast.expressions import (
    PsAnd,
    PsOr,
    PsNot,
    PsEq,
    PsGt,
    PsTernary,
    PsRem,
    PsIntDiv
)

from pystencils.types.quick import Int, Fp, Bool

x, y, z = [PsExpression.make(PsSymbol(name, Fp(32))) for name in "xyz"]
p, q, r = [PsExpression.make(PsSymbol(name, Int(32))) for name in "pqr"]
a, b, c = [PsExpression.make(PsSymbol(name, Bool())) for name in "abc"]

f3p5 = PsExpression.make(PsConstant(3.5, Fp(32)))
f42 = PsExpression.make(PsConstant(42, Fp(32)))

f0 = PsExpression.make(PsConstant(0.0, Fp(32)))
f1 = PsExpression.make(PsConstant(1.0, Fp(32)))

i0 = PsExpression.make(PsConstant(0, Int(32)))
i1 = PsExpression.make(PsConstant(1, Int(32)))
im1 = PsExpression.make(PsConstant(-1, Int(32)))

i3 = PsExpression.make(PsConstant(3, Int(32)))
i4 = PsExpression.make(PsConstant(4, Int(32)))
im3 = PsExpression.make(PsConstant(-3, Int(32)))
i12 = PsExpression.make(PsConstant(12, Int(32)))

true = PsExpression.make(PsConstant(True, Bool()))
false = PsExpression.make(PsConstant(False, Bool()))


def test_idempotence():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify(f42 * (f1 + f0) - f0)
    result = elim(expr)
    assert isinstance(result, PsConstantExpr) and result.structurally_equal(f42)

    expr = typify((x + f0) * f3p5 + (f1 * y + f0) * f42)
    result = elim(expr)
    assert result.structurally_equal(x * f3p5 + y * f42)

    expr = typify((f3p5 * f1) + (f42 * f1))
    result = elim(expr)
    #   do not fold floats by default
    assert expr.structurally_equal(f3p5 + f42)

    expr = typify(f1 * x + f0 + (f0 + f0 + f1 + f0) * y)
    result = elim(expr)
    assert result.structurally_equal(x + y)


def test_int_folding():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify((i1 * p + i1 * -i3) + i1 * i12)
    result = elim(expr)
    assert result.structurally_equal((p + im3) + i12)

    expr = typify((i1 + i1 + i1 + i0 + i0 + i1) * (i1 + i1 + i1))
    result = elim(expr)
    assert result.structurally_equal(i12)


def test_zero_dominance():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify((f0 * x) + (y * f0) + f1)
    result = elim(expr)
    assert result.structurally_equal(f1)

    expr = typify((i3 + i12 * (p + q) + p / (i3 * q)) * i0)
    result = elim(expr)
    assert result.structurally_equal(i0)


def test_divisions():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify(f3p5 / f1)
    result = elim(expr)
    assert result.structurally_equal(f3p5)

    expr = typify(i3 / i1)
    result = elim(expr)
    assert result.structurally_equal(i3)

    expr = typify(PsRem(i3, i1))
    result = elim(expr)
    assert result.structurally_equal(i0)

    expr = typify(PsIntDiv(i12, i3))
    result = elim(expr)
    assert result.structurally_equal(i4)

    expr = typify(i12 / i3)
    result = elim(expr)
    assert result.structurally_equal(i4)

    expr = typify(PsIntDiv(i4, i3))
    result = elim(expr)
    assert result.structurally_equal(i1)

    expr = typify(PsIntDiv(-i4, i3))
    result = elim(expr)
    assert result.structurally_equal(im1)

    expr = typify(PsIntDiv(i4, -i3))
    result = elim(expr)
    assert result.structurally_equal(im1)

    expr = typify(PsIntDiv(-i4, -i3))
    result = elim(expr)
    assert result.structurally_equal(i1)

    expr = typify(PsRem(i4, i3))
    result = elim(expr)
    assert result.structurally_equal(i1)

    expr = typify(PsRem(-i4, i3))
    result = elim(expr)
    assert result.structurally_equal(im1)

    expr = typify(PsRem(i4, -i3))
    result = elim(expr)
    assert result.structurally_equal(i1)

    expr = typify(PsRem(-i4, -i3))
    result = elim(expr)
    assert result.structurally_equal(im1)


def test_boolean_folding():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify(PsNot(PsAnd(false, PsOr(true, a))))
    result = elim(expr)
    assert result.structurally_equal(true)

    expr = typify(PsOr(PsAnd(a, b), PsNot(false)))
    result = elim(expr)
    assert result.structurally_equal(true)

    expr = typify(PsAnd(c, PsAnd(true, PsAnd(a, PsOr(false, b)))))
    result = elim(expr)
    assert result.structurally_equal(PsAnd(c, PsAnd(a, b)))


def test_relations_folding():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify(PsGt(p * i0, - i1))
    result = elim(expr)
    assert result.structurally_equal(true)

    expr = typify(PsEq(i1 + i1 + i1, i3))
    result = elim(expr)
    assert result.structurally_equal(true)

    expr = typify(PsEq(- i1, - i3))
    result = elim(expr)
    assert result.structurally_equal(false)

    expr = typify(PsEq(x + y, f1 * (x + y)))
    result = elim(expr)
    assert result.structurally_equal(true)

    expr = typify(PsGt(x + y, f1 * (x + y)))
    result = elim(expr)
    assert result.structurally_equal(false)


def test_ternary_folding():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateConstants(ctx)

    expr = typify(PsTernary(true, x, y))
    result = elim(expr)
    assert result.structurally_equal(x)

    expr = typify(PsTernary(false, x, y))
    result = elim(expr)
    assert result.structurally_equal(y)

    expr = typify(PsTernary(PsGt(i1, i0), PsTernary(PsEq(i1, i12), x, y), z))
    result = elim(expr)
    assert result.structurally_equal(y)

    expr = typify(PsTernary(PsGt(x, y), x + f0, y * f1))
    result = elim(expr)
    assert result.structurally_equal(PsTernary(PsGt(x, y), x, y))
