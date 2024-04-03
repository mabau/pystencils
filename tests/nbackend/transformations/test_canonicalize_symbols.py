#  type: ignore
import sympy as sp

from pystencils import Field, Assignment, AddAugmentedAssignment, make_slice, DEFAULTS

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
)
from pystencils.backend.transformations import CanonicalizeSymbols
from pystencils.backend.ast.structural import PsConditional, PsBlock


def test_deduplication():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    canonicalize = CanonicalizeSymbols(ctx)

    f = Field.create_fixed_size("f", (5, 5), strides=(5, 1))
    x, y, z = sp.symbols("x, y, z")

    ispace = FullIterationSpace.create_from_slice(ctx, make_slice[:, :], f)
    ctx.set_iteration_space(ispace)

    ctr_1 = DEFAULTS.spatial_counters[1]

    then_branch = PsBlock(
        [
            factory.parse_sympy(Assignment(x, y)),
            factory.parse_sympy(Assignment(f.center(0), x)),
        ]
    )

    else_branch = PsBlock(
        [
            factory.parse_sympy(Assignment(x, z)),
            factory.parse_sympy(Assignment(f.center(0), x)),
        ]
    )

    ast = PsConditional(
        factory.parse_sympy(ctr_1),
        then_branch,
        else_branch,
    )

    ast = factory.loops_from_ispace(ispace, PsBlock([ast]))

    ast = canonicalize(ast)

    assert canonicalize.get_last_live_symbols() == {
        ctx.find_symbol("y"),
        ctx.find_symbol("z"),
        ctx.get_array(f).base_pointer,
    }

    assert ctx.find_symbol("x") is not None
    assert ctx.find_symbol("x__0") is not None

    assert then_branch.statements[0].declared_symbol.name == "x__0"
    assert then_branch.statements[1].rhs.symbol.name == "x__0"

    assert else_branch.statements[0].declared_symbol.name == "x"
    assert else_branch.statements[1].rhs.symbol.name == "x"

    assert ctx.find_symbol("x").dtype.const
    assert ctx.find_symbol("x__0").dtype.const
    assert ctx.find_symbol("y").dtype.const
    assert ctx.find_symbol("z").dtype.const


def test_do_not_constify():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    canonicalize = CanonicalizeSymbols(ctx)

    x, z = sp.symbols("x, z")

    ast = factory.loop("i", make_slice[:10], PsBlock([
        factory.parse_sympy(Assignment(x, z)),
        factory.parse_sympy(AddAugmentedAssignment(z, 1))
    ]))

    ast = canonicalize(ast)

    assert ctx.find_symbol("x").dtype.const
    assert not ctx.find_symbol("z").dtype.const
