import sympy as sp

from pystencils import Assignment, fields

from pystencils.backend.ast.structural import (
    PsAssignment,
    PsDeclaration,
)
from pystencils.backend.ast.expressions import (
    PsExpression,
    PsArrayAccess
)
from pystencils.backend.constants import PsConstant
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    FreezeExpressions,
    FullIterationSpace,
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

    lhs = PsArrayAccess(f_arr.base_pointer, (PsExpression.make(counter) + zero) * PsExpression.make(f_arr.strides[0]) + zero * one)
    rhs = PsArrayAccess(g_arr.base_pointer, (PsExpression.make(counter) + zero) * PsExpression.make(g_arr.strides[0]) + zero * one)

    should = PsAssignment(lhs, rhs)

    assert fasm.structurally_equal(should)
