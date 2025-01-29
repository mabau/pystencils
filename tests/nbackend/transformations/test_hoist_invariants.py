import sympy as sp

from pystencils import (
    Field,
    TypedSymbol,
    Assignment,
    AddAugmentedAssignment,
    make_slice,
)
from pystencils.types.quick import Arr, Fp, Bool

from pystencils.backend.ast.structural import (
    PsBlock,
    PsLoop,
    PsConditional,
    PsDeclaration,
)

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
)
from pystencils.backend.transformations import (
    CanonicalizeSymbols,
    HoistLoopInvariantDeclarations,
)


def test_hoist_multiple_loops():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    canonicalize = CanonicalizeSymbols(ctx)
    hoist = HoistLoopInvariantDeclarations(ctx)

    f = Field.create_fixed_size("f", (5, 5), memory_strides=(5, 1))
    x, y, z = sp.symbols("x, y, z")

    ispace = FullIterationSpace.create_from_slice(ctx, make_slice[:, :], f)
    ctx.set_iteration_space(ispace)

    first_loop = factory.loops_from_ispace(
        ispace,
        PsBlock(
            [
                factory.parse_sympy(Assignment(x, y)),
                factory.parse_sympy(Assignment(f.center(0), x)),
            ]
        ),
    )

    second_loop = factory.loops_from_ispace(
        ispace,
        PsBlock(
            [
                factory.parse_sympy(Assignment(x, z)),
                factory.parse_sympy(Assignment(f.center(0), x)),
            ]
        ),
    )

    ast = PsBlock([first_loop, second_loop])

    result = canonicalize(ast)
    result = hoist(result)

    assert isinstance(result, PsBlock)

    assert (
        isinstance(result.statements[0], PsDeclaration)
        and result.statements[0].declared_symbol.name == "x__0"
    )

    assert result.statements[1] == first_loop

    assert (
        isinstance(result.statements[2], PsDeclaration)
        and result.statements[2].declared_symbol.name == "x"
    )

    assert result.statements[3] == second_loop


def test_hoist_with_recurrence():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    hoist = HoistLoopInvariantDeclarations(ctx)

    x, y = sp.symbols("x, y")
    x_decl = factory.parse_sympy(Assignment(x, 1))
    x_update = factory.parse_sympy(AddAugmentedAssignment(x, 1))
    y_decl = factory.parse_sympy(Assignment(y, 2 * x))

    loop = factory.loop("i", make_slice[0:10:1], PsBlock([y_decl, x_update]))

    ast = PsBlock([x_decl, loop])

    result = hoist(ast)

    #   x is updated in the loop, so nothing can be hoisted
    assert isinstance(result, PsBlock)
    assert result.statements == [x_decl, loop]
    assert loop.body.statements == [y_decl, x_update]


def test_hoist_with_conditionals():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    hoist = HoistLoopInvariantDeclarations(ctx)

    x, y, z, w = sp.symbols("x, y, z, w")
    x_decl = factory.parse_sympy(Assignment(x, 1))
    x_update = factory.parse_sympy(AddAugmentedAssignment(x, 1))
    y_decl = factory.parse_sympy(Assignment(y, 2 * x))
    z_decl = factory.parse_sympy(Assignment(z, 312))
    w_decl = factory.parse_sympy(Assignment(w, 142))

    cond = factory.parse_sympy(TypedSymbol("cond", Bool()))

    inner_conditional = PsConditional(cond, PsBlock([x_update, z_decl]))
    loop = factory.loop(
        "i",
        make_slice[0:10:1],
        PsBlock([y_decl, w_decl, inner_conditional]),
    )
    outer_conditional = PsConditional(cond, PsBlock([loop]))

    ast = PsBlock([x_decl, outer_conditional])

    result = hoist(ast)

    #   z is hidden inside conditional, so z cannot be hoisted
    #   x is updated conditionally, so y cannot be hoisted
    assert isinstance(result, PsBlock)
    assert result.statements == [x_decl, outer_conditional]
    assert outer_conditional.branch_true.statements == [w_decl, loop]
    assert loop.body.statements == [y_decl, inner_conditional]


def test_hoist_arrays():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    hoist = HoistLoopInvariantDeclarations(ctx)

    const_arr_symb = TypedSymbol(
        "const_arr",
        Arr(Fp(64), (10,), const=True),
    )
    const_array_decl = factory.parse_sympy(Assignment(const_arr_symb, tuple(range(10))))
    const_arr = sp.IndexedBase(const_arr_symb, shape=(10,))

    arr_symb = TypedSymbol(
        "arr",
        Arr(Fp(64), (10,), const=False),
    )
    array_decl = factory.parse_sympy(Assignment(arr_symb, tuple(range(10))))
    arr = sp.IndexedBase(arr_symb, shape=(10,))

    x, y = sp.symbols("x, y")

    nonconst_usage = factory.parse_sympy(Assignment(x, arr[3]))
    const_usage = factory.parse_sympy(Assignment(y, const_arr[3]))
    body = PsBlock([array_decl, const_array_decl, nonconst_usage, const_usage])

    loop = factory.loop_nest(("i", "j"), make_slice[:10, :42], body)

    result = hoist(loop)

    assert isinstance(result, PsBlock)
    assert result.statements == [array_decl, const_array_decl, const_usage, loop]

    assert isinstance(loop.body.statements[0], PsLoop)
    assert loop.body.statements[0].body.statements == [nonconst_usage]


def test_hoisting_eliminates_loops():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    hoist = HoistLoopInvariantDeclarations(ctx)

    x, y, z = sp.symbols("x, y, z")

    invariant_decls = [
        factory.parse_sympy(Assignment(x, 42)),
        factory.parse_sympy(Assignment(y, 2 * x)),
        factory.parse_sympy(Assignment(z, x + 4 * y)),
    ]

    ast = factory.loop_nest(("i", "j"), make_slice[:10, :42], PsBlock(invariant_decls))

    ast = hoist(ast)

    assert isinstance(ast, PsBlock)
    #   All statements are hoisted and the loops are removed
    assert ast.statements == invariant_decls


def test_hoist_mutation():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    hoist = HoistLoopInvariantDeclarations(ctx)

    x = sp.Symbol("x")
    x_decl = factory.parse_sympy(Assignment(x, 1))
    x_update = factory.parse_sympy(AddAugmentedAssignment(x, 1))

    inner_loop = factory.loop("j", slice(10), PsBlock([x_update]))
    outer_loop = factory.loop("i", slice(10), PsBlock([x_decl, inner_loop]))

    result = hoist(outer_loop)

    #   x is updated in the loop, so nothing can be hoisted
    assert isinstance(result, PsLoop)
    assert result.body.statements == [x_decl, inner_loop]
