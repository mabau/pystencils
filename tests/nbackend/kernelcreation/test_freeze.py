import sympy as sp

from pystencils import Assignment, fields

from pystencils.backend.ast.structural import (
    PsAssignment,
    PsBlock,
    PsDeclaration,
)
from pystencils.backend.ast.expressions import (
    PsArrayAccess,
    PsBitwiseAnd,
    PsBitwiseOr,
    PsBitwiseXor,
    PsExpression,
    PsIntDiv,
    PsLeftShift,
    PsMul,
    PsRightShift,
)
from pystencils.backend.constants import PsConstant
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    FreezeExpressions,
    FullIterationSpace,
)

from pystencils.sympyextensions.integer_functions import (
    bit_shift_left,
    bit_shift_right,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    int_div,
    int_power_of_2,
    modulo_floor,
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
