from pystencils import make_slice
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    Typifier,
    AstFactory,
)
from pystencils.backend.ast.expressions import (
    PsExpression,
    PsEq,
    PsGe,
    PsGt,
    PsLe,
    PsLt,
)
from pystencils.backend.ast.structural import PsConditional, PsBlock, PsComment
from pystencils.backend.constants import PsConstant
from pystencils.backend.transformations import EliminateBranches
from pystencils.types.quick import Int


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


def test_isl():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    typify = Typifier(ctx)
    elim = EliminateBranches(ctx)

    i = PsExpression.make(ctx.get_symbol("i", ctx.index_dtype))
    j = PsExpression.make(ctx.get_symbol("j", ctx.index_dtype))

    const_2 = PsExpression.make(PsConstant(2, ctx.index_dtype))
    const_4 = PsExpression.make(PsConstant(4, ctx.index_dtype))

    a_true = PsBlock([PsComment("a true")])
    a_false = PsBlock([PsComment("a false")])
    b_true = PsBlock([PsComment("b true")])
    b_false = PsBlock([PsComment("b false")])
    c_true = PsBlock([PsComment("c true")])
    c_false = PsBlock([PsComment("c false")])

    a = PsConditional(PsLt(i + j, const_2 * const_4), a_true, a_false)
    b = PsConditional(PsGe(j, const_4), b_true, b_false)
    c = PsConditional(PsEq(i, const_4), c_true, c_false)

    outer_loop = factory.loop(j.symbol.name, slice(0, 3), PsBlock([a, b, c]))
    outer_cond = typify(
        PsConditional(PsLe(i, const_4), PsBlock([outer_loop]), PsBlock([]))
    )
    ast = outer_cond

    result = elim(ast)

    assert result.branch_true.statements[0].body.statements[0] == a_true
    assert result.branch_true.statements[0].body.statements[1] == b_false
    assert result.branch_true.statements[0].body.statements[2] == c
