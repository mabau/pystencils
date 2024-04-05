from pystencils import make_slice
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    Typifier,
    AstFactory,
)
from pystencils.backend.ast.expressions import PsExpression
from pystencils.backend.ast.structural import PsConditional, PsBlock, PsComment
from pystencils.backend.constants import PsConstant
from pystencils.backend.transformations import EliminateBranches
from pystencils.types.quick import Int
from pystencils.backend.ast.expressions import PsGt


i0 = PsExpression.make(PsConstant(0, Int(32)))
i1 = PsExpression.make(PsConstant(1, Int(32)))


def test_eliminate_conditional():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)
    elim = EliminateBranches(ctx)

    b1 = PsBlock([PsComment("Branch One")])

    b2 = PsBlock([PsComment("Branch Two")])

    cond = typify(PsConditional(PsGt(i1, i0), b1, b2))
    result = elim(cond)
    assert result == b1

    cond = typify(PsConditional(PsGt(-i1, i0), b1, b2))
    result = elim(cond)
    assert result == b2

    cond = typify(PsConditional(PsGt(-i1, i0), b1))
    result = elim(cond)
    assert result.structurally_equal(PsBlock([]))


def test_eliminate_nested_conditional():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    typify = Typifier(ctx)
    elim = EliminateBranches(ctx)

    b1 = PsBlock([PsComment("Branch One")])

    b2 = PsBlock([PsComment("Branch Two")])

    cond = typify(PsConditional(PsGt(i1, i0), b1, b2))
    ast = factory.loop_nest(("i", "j"), make_slice[:10, :10], PsBlock([cond]))

    result = elim(ast)
    assert result.body.statements[0].body.statements[0] == b1
