import sympy as sp
import pymbolic.primitives as pb

from pystencils import Assignment, fields

from pystencils.nbackend.ast import (
    PsAssignment,
    PsDeclaration,
    PsExpression,
    PsSymbolExpr,
    PsLvalueExpr,
)
from pystencils.nbackend.typed_expressions import PsTypedConstant, PsTypedVariable
from pystencils.nbackend.arrays import PsArrayAccess
from pystencils.nbackend.kernelcreation import (
    KernelCreationOptions,
    KernelCreationContext,
    FreezeExpressions,
    FullIterationSpace,
)


def test_freeze_simple():
    options = KernelCreationOptions()
    ctx = KernelCreationContext(options)
    freeze = FreezeExpressions(ctx)

    x, y, z = sp.symbols("x, y, z")
    asm = Assignment(z, 2 * x + y)

    fasm = freeze(asm)

    pb_x, pb_y, pb_z = pb.variables("x y z")

    assert fasm == PsDeclaration(PsSymbolExpr(pb_z), PsExpression(pb_y + 2 * pb_x))
    assert fasm != PsAssignment(PsSymbolExpr(pb_z), PsExpression(pb_y + 2 * pb_x))


def test_freeze_fields():
    options = KernelCreationOptions()
    ctx = KernelCreationContext(options)

    zero = PsTypedConstant(0, ctx.index_dtype)
    forty_two = PsTypedConstant(42, ctx.index_dtype)
    one = PsTypedConstant(1, ctx.index_dtype)
    counter = PsTypedVariable("ctr", ctx.index_dtype)
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

    lhs = PsArrayAccess(f_arr.base_pointer, pb.Sum((counter * f_arr.strides[0], zero)))
    rhs = PsArrayAccess(g_arr.base_pointer, pb.Sum((counter * g_arr.strides[0], zero)))

    should = PsAssignment(PsLvalueExpr(lhs), PsExpression(rhs))

    assert fasm == should
