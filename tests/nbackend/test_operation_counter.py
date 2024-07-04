from pystencils.backend.ast.analysis import OperationCounter
from pystencils.backend.ast.expressions import (
    PsAdd,
    PsConstant,
    PsDiv,
    PsExpression,
    PsMul,
    PsTernary,
)
from pystencils.backend.ast.structural import (
    PsBlock,
    PsDeclaration,
    PsLoop,
)

from pystencils.backend.kernelcreation import KernelCreationContext, Typifier
from pystencils.types import PsBoolType


def test_count_operations():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    counter = OperationCounter()

    x = PsExpression.make(ctx.get_symbol("x"))
    y = PsExpression.make(ctx.get_symbol("y"))
    z = PsExpression.make(ctx.get_symbol("z"))

    i = PsExpression.make(ctx.get_symbol("i", ctx.index_dtype))
    p = PsExpression.make(ctx.get_symbol("p", PsBoolType()))

    zero = PsExpression.make(PsConstant(0, ctx.index_dtype))
    two = PsExpression.make(PsConstant(2, ctx.index_dtype))
    five = PsExpression.make(PsConstant(5, ctx.index_dtype))

    ast = PsLoop(
        i,
        zero,
        five,
        two,
        PsBlock(
            [
                PsDeclaration(x, PsAdd(y, z)),
                PsDeclaration(y, PsMul(x, PsMul(y, z))),
                PsDeclaration(z, PsDiv(PsDiv(PsDiv(x, y), z), PsTernary(p, x, y))),
            ]
        ),
    )
    ast = typify(ast)

    op_count = counter(ast)

    assert op_count.float_adds == 3 * 1
    assert op_count.float_muls == 3 * 2
    assert op_count.float_divs == 3 * 3
    assert op_count.int_adds == 3 * 1
    assert op_count.int_muls == 0
    assert op_count.int_divs == 0
    assert op_count.calls == 0
    assert op_count.branches == 3 * 1
    assert op_count.loops_with_dynamic_bounds == 0
