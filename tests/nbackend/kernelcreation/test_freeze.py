import sympy as sp
import pytest

from pystencils import Assignment, fields, create_type, create_numeric_type, TypedSymbol, DynamicType
from pystencils.sympyextensions import CastFunc

from pystencils.backend.ast.structural import (
    PsAssignment,
    PsDeclaration,
)
from pystencils.backend.ast.expressions import (
    PsArrayAccess,
    PsBitwiseAnd,
    PsBitwiseOr,
    PsBitwiseXor,
    PsExpression,
    PsTernary,
    PsIntDiv,
    PsLeftShift,
    PsRightShift,
    PsAnd,
    PsOr,
    PsNot,
    PsEq,
    PsNe,
    PsLt,
    PsLe,
    PsGt,
    PsGe,
    PsCall,
    PsCast,
)
from pystencils.backend.constants import PsConstant
from pystencils.backend.functions import PsMathFunction, MathFunctions
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    FreezeExpressions,
    FullIterationSpace,
)
from pystencils.backend.kernelcreation.freeze import FreezeError

from pystencils.sympyextensions.integer_functions import (
    bit_shift_left,
    bit_shift_right,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    int_div,
    int_power_of_2,
)


def test_freeze_simple():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)

    x, y, z = sp.symbols("x, y, z")
    asm = Assignment(z, 2 * x + y)

    fasm = freeze(asm)

    x2 = PsExpression.make(ctx.get_symbol("x"))
    y2 = PsExpression.make(ctx.get_symbol("y"))
    z2 = PsExpression.make(ctx.get_symbol("z"))

    two = PsExpression.make(PsConstant(2))

    should = PsDeclaration(z2, y2 + two * x2)

    assert fasm.structurally_equal(should)
    assert not fasm.structurally_equal(PsAssignment(z2, two * x2 + y2))


def test_freeze_fields():
    ctx = KernelCreationContext()

    zero = PsExpression.make(PsConstant(0, ctx.index_dtype))
    forty_two = PsExpression.make(PsConstant(42, ctx.index_dtype))
    one = PsExpression.make(PsConstant(1, ctx.index_dtype))
    counter = ctx.get_symbol("ctr", ctx.index_dtype)
    ispace = FullIterationSpace(
        ctx, [FullIterationSpace.Dimension(zero, forty_two, one, counter)]
    )
    ctx.set_iteration_space(ispace)

    freeze = FreezeExpressions(ctx)

    f, g = fields("f, g : [1D]")
    asm = Assignment(f.center(0), g.center(0))

    f_arr = ctx.get_array(f)
    g_arr = ctx.get_array(g)

    fasm = freeze(asm)

    zero = PsExpression.make(PsConstant(0))

    lhs = PsArrayAccess(
        f_arr.base_pointer,
        (PsExpression.make(counter) + zero) * PsExpression.make(f_arr.strides[0])
        + zero * one,
    )
    rhs = PsArrayAccess(
        g_arr.base_pointer,
        (PsExpression.make(counter) + zero) * PsExpression.make(g_arr.strides[0])
        + zero * one,
    )

    should = PsAssignment(lhs, rhs)

    assert fasm.structurally_equal(should)


def test_freeze_integer_binops():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)

    x, y, z = sp.symbols("x, y, z")
    expr = bit_shift_left(
        bit_shift_right(bitwise_and(x, y), bitwise_or(y, z)), bitwise_xor(x, z)
    )

    fexpr = freeze(expr)

    x2 = PsExpression.make(ctx.get_symbol("x"))
    y2 = PsExpression.make(ctx.get_symbol("y"))
    z2 = PsExpression.make(ctx.get_symbol("z"))

    should = PsLeftShift(
        PsRightShift(PsBitwiseAnd(x2, y2), PsBitwiseOr(y2, z2)), PsBitwiseXor(x2, z2)
    )

    assert fexpr.structurally_equal(should)


def test_freeze_integer_functions():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)

    x2 = PsExpression.make(ctx.get_symbol("x", ctx.index_dtype))
    y2 = PsExpression.make(ctx.get_symbol("y", ctx.index_dtype))
    z2 = PsExpression.make(ctx.get_symbol("z", ctx.index_dtype))

    x, y, z = sp.symbols("x, y, z")
    asms = [
        Assignment(z, int_div(x, y)),
        Assignment(z, int_power_of_2(x, y)),
        # Assignment(z, modulo_floor(x, y)),
    ]

    fasms = [freeze(asm) for asm in asms]

    should = [
        PsDeclaration(z2, PsIntDiv(x2, y2)),
        PsDeclaration(z2, PsLeftShift(PsExpression.make(PsConstant(1)), x2)),
        # PsDeclaration(z2, PsMul(PsIntDiv(x2, y2), y2)),
    ]

    for fasm, correct in zip(fasms, should):
        assert fasm.structurally_equal(correct)


def test_freeze_booleans():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)

    x2 = PsExpression.make(ctx.get_symbol("x"))
    y2 = PsExpression.make(ctx.get_symbol("y"))
    z2 = PsExpression.make(ctx.get_symbol("z"))
    w2 = PsExpression.make(ctx.get_symbol("w"))

    x, y, z, w = sp.symbols("x, y, z, w")

    expr = freeze(sp.Not(sp.And(x, y)))
    assert expr.structurally_equal(PsNot(PsAnd(x2, y2)))

    expr = freeze(sp.Or(sp.Not(z), sp.And(y, sp.Not(x))))
    assert expr.structurally_equal(PsOr(PsNot(z2), PsAnd(y2, PsNot(x2))))

    expr = freeze(sp.And(w, x, y, z))
    assert expr.structurally_equal(PsAnd(PsAnd(PsAnd(w2, x2), y2), z2))

    expr = freeze(sp.Or(w, x, y, z))
    assert expr.structurally_equal(PsOr(PsOr(PsOr(w2, x2), y2), z2))


@pytest.mark.parametrize(
    "rel_pair",
    [
        (sp.Eq, PsEq),
        (sp.Ne, PsNe),
        (sp.Lt, PsLt),
        (sp.Gt, PsGt),
        (sp.Le, PsLe),
        (sp.Ge, PsGe),
    ],
)
def test_freeze_relations(rel_pair):
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)

    sp_op, ps_op = rel_pair

    x2 = PsExpression.make(ctx.get_symbol("x"))
    y2 = PsExpression.make(ctx.get_symbol("y"))
    z2 = PsExpression.make(ctx.get_symbol("z"))

    x, y, z = sp.symbols("x, y, z")

    expr1 = freeze(sp_op(x, y + z))
    assert expr1.structurally_equal(ps_op(x2, y2 + z2))


def test_freeze_piecewise():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)

    p, q, x, y, z = sp.symbols("p, q, x, y, z")

    p2 = PsExpression.make(ctx.get_symbol("p"))
    q2 = PsExpression.make(ctx.get_symbol("q"))
    x2 = PsExpression.make(ctx.get_symbol("x"))
    y2 = PsExpression.make(ctx.get_symbol("y"))
    z2 = PsExpression.make(ctx.get_symbol("z"))

    piecewise = sp.Piecewise((x, p), (y, q), (z, True))
    expr = freeze(piecewise)

    assert isinstance(expr, PsTernary)

    should = PsTernary(p2, x2, PsTernary(q2, y2, z2))
    assert expr.structurally_equal(should)

    piecewise = sp.Piecewise((x, p), (y, q), (z, sp.Or(p, q)))
    with pytest.raises(FreezeError):
        freeze(piecewise)


def test_multiarg_min_max():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)

    w, x, y, z = sp.symbols("w, x, y, z")

    x2 = PsExpression.make(ctx.get_symbol("x"))
    y2 = PsExpression.make(ctx.get_symbol("y"))
    z2 = PsExpression.make(ctx.get_symbol("z"))
    w2 = PsExpression.make(ctx.get_symbol("w"))

    def op(a, b):
        return PsCall(PsMathFunction(MathFunctions.Min), (a, b))

    expr = freeze(sp.Min(w, x, y))
    assert expr.structurally_equal(op(op(w2, x2), y2))

    expr = freeze(sp.Min(w, x, y, z))
    assert expr.structurally_equal(op(op(w2, x2), op(y2, z2)))

    def op(a, b):
        return PsCall(PsMathFunction(MathFunctions.Max), (a, b))

    expr = freeze(sp.Max(w, x, y))
    assert expr.structurally_equal(op(op(w2, x2), y2))

    expr = freeze(sp.Max(w, x, y, z))
    assert expr.structurally_equal(op(op(w2, x2), op(y2, z2)))


def test_dynamic_types():
    ctx = KernelCreationContext(
        default_dtype=create_numeric_type("float16"), index_dtype=create_type("int16")
    )
    freeze = FreezeExpressions(ctx)

    x, y = [TypedSymbol(n, DynamicType.NUMERIC_TYPE) for n in "xy"]
    p, q = [TypedSymbol(n, DynamicType.INDEX_TYPE) for n in "pq"]

    expr = freeze(x + y)
    
    assert ctx.get_symbol("x").dtype == ctx.default_dtype
    assert ctx.get_symbol("y").dtype == ctx.default_dtype

    expr = freeze(p - q)    
    assert ctx.get_symbol("p").dtype == ctx.index_dtype
    assert ctx.get_symbol("q").dtype == ctx.index_dtype


def test_cast_func():
    ctx = KernelCreationContext(
        default_dtype=create_numeric_type("float16"), index_dtype=create_type("int16")
    )
    freeze = FreezeExpressions(ctx)

    x, y, z = sp.symbols("x, y, z")

    x2 = PsExpression.make(ctx.get_symbol("x"))
    y2 = PsExpression.make(ctx.get_symbol("y"))
    z2 = PsExpression.make(ctx.get_symbol("z"))

    expr = freeze(CastFunc(x, create_type("int")))
    assert expr.structurally_equal(PsCast(create_type("int"), x2))

    expr = freeze(CastFunc.as_numeric(y))
    assert expr.structurally_equal(PsCast(ctx.default_dtype, y2))

    expr = freeze(CastFunc.as_index(z))
    assert expr.structurally_equal(PsCast(ctx.index_dtype, z2))
