import sympy as sp

from pystencils import Field, Assignment, make_slice
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
)
from pystencils.backend.transformations import ReshapeLoops

from pystencils.backend.ast.structural import (
    PsDeclaration,
    PsBlock,
    PsLoop,
    PsConditional,
)
from pystencils.backend.ast.expressions import PsConstantExpr, PsGe, PsLt


def test_loop_cutting():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    reshape = ReshapeLoops(ctx)

    x, y, z = sp.symbols("x, y, z")

    f = Field.create_generic("f", 1, index_shape=(2,))
    ispace = FullIterationSpace.create_from_slice(ctx, make_slice[:], archetype_field=f)
    ctx.set_iteration_space(ispace)

    loop_body = PsBlock(
        [
            factory.parse_sympy(Assignment(x, 2 * z)),
            factory.parse_sympy(Assignment(f.center(0), x + y)),
        ]
    )

    loop = factory.loops_from_ispace(ispace, loop_body)

    subloops = reshape.cut_loop(loop, [1, 1, 3])
    assert len(subloops) == 3

    subloop = subloops[0]
    assert isinstance(subloop, PsBlock)
    assert isinstance(subloop.statements[0], PsDeclaration)
    assert subloop.statements[0].declared_symbol.name == "ctr_0__0"

    x_decl = subloop.statements[1]
    assert isinstance(x_decl, PsDeclaration)
    assert x_decl.declared_symbol.name == "x__0"

    subloop = subloops[1]
    assert isinstance(subloop, PsLoop)
    assert (
        isinstance(subloop.start, PsConstantExpr) and subloop.start.constant.value == 1
    )
    assert isinstance(subloop.stop, PsConstantExpr) and subloop.stop.constant.value == 3

    x_decl = subloop.body.statements[0]
    assert isinstance(x_decl, PsDeclaration)
    assert x_decl.declared_symbol.name == "x__1"

    subloop = subloops[2]
    assert isinstance(subloop, PsLoop)
    assert (
        isinstance(subloop.start, PsConstantExpr) and subloop.start.constant.value == 3
    )
    assert subloop.stop.structurally_equal(loop.stop)


def test_loop_peeling():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    reshape = ReshapeLoops(ctx)

    x, y, z = sp.symbols("x, y, z")

    f = Field.create_generic("f", 1, index_shape=(2,))
    ispace = FullIterationSpace.create_from_slice(
        ctx, slice(2, 11, 3), archetype_field=f
    )
    ctx.set_iteration_space(ispace)

    loop_body = PsBlock(
        [
            factory.parse_sympy(Assignment(x, 2 * z)),
            factory.parse_sympy(Assignment(f.center(0), x + y)),
        ]
    )

    loop = factory.loops_from_ispace(ispace, loop_body)

    num_iters = 2
    peeled_iters, peeled_loop = reshape.peel_loop_front(loop, num_iters)
    assert len(peeled_iters) == num_iters

    for i, iter in enumerate(peeled_iters):
        assert isinstance(iter, PsBlock)

        ctr_decl = iter.statements[0]
        assert isinstance(ctr_decl, PsDeclaration)
        assert ctr_decl.declared_symbol.name == f"ctr_0__{i}"
        ctr_value = {0: 2, 1: 5}[i]
        assert ctr_decl.rhs.structurally_equal(factory.parse_index(ctr_value))

        cond = iter.statements[1]
        assert isinstance(cond, PsConditional)
        assert cond.condition.structurally_equal(PsLt(ctr_decl.lhs, loop.stop))

        subblock = cond.branch_true
        assert isinstance(subblock.statements[0], PsDeclaration)
        assert subblock.statements[0].declared_symbol.name == f"x__{i}"

    assert peeled_loop.start.structurally_equal(factory.parse_index(8))
    assert peeled_loop.stop.structurally_equal(loop.stop)
    assert peeled_loop.body.structurally_equal(loop.body)


def test_loop_peeling_back():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    reshape = ReshapeLoops(ctx)

    x, y, z = sp.symbols("x, y, z")

    f = Field.create_generic("f", 1, index_shape=(2,))
    ispace = FullIterationSpace.create_from_slice(ctx, make_slice[:], archetype_field=f)
    ctx.set_iteration_space(ispace)

    loop_body = PsBlock(
        [
            factory.parse_sympy(Assignment(x, 2 * z)),
            factory.parse_sympy(Assignment(f.center(0), x + y)),
        ]
    )

    loop = factory.loops_from_ispace(ispace, loop_body)

    num_iters = 3
    peeled_loop, peeled_iters = reshape.peel_loop_back(loop, num_iters)
    assert len(peeled_iters) == 3

    for i, iter in enumerate(peeled_iters):
        assert isinstance(iter, PsBlock)

        ctr_decl = iter.statements[0]
        assert isinstance(ctr_decl, PsDeclaration)
        assert ctr_decl.declared_symbol.name == f"ctr_0__{i}"

        cond = iter.statements[1]
        assert isinstance(cond, PsConditional)
        assert cond.condition.structurally_equal(PsGe(ctr_decl.lhs, loop.start))

        subblock = cond.branch_true
        assert isinstance(subblock.statements[0], PsDeclaration)
        assert subblock.statements[0].declared_symbol.name == f"x__{i}"

    assert peeled_loop.start.structurally_equal(loop.start)
    assert peeled_loop.stop.structurally_equal(
        factory.loops_from_ispace(ispace, loop_body).stop
        - factory.parse_index(num_iters)
    )
    assert peeled_loop.body.structurally_equal(loop.body)
