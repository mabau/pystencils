import sympy as sp
from pystencils import Field, Assignment, make_slice, TypedSymbol
from pystencils.types.quick import Arr

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
)
from pystencils.backend.transformations import CanonicalClone
from pystencils.backend.ast.structural import PsBlock, PsComment
from pystencils.backend.ast.expressions import PsSymbolExpr
from pystencils.backend.ast.iteration import dfs_preorder


def test_clone_entire_ast():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    canon_clone = CanonicalClone(ctx)

    f = Field.create_generic("f", 2, index_shape=(5,))
    rho = sp.Symbol("rho")
    u = sp.symbols("u_:2")

    cx = TypedSymbol("cx", Arr(ctx.default_dtype, (5,)))
    cy = TypedSymbol("cy", Arr(ctx.default_dtype, (5,)))
    cxs = sp.IndexedBase(cx, shape=(5,))
    cys = sp.IndexedBase(cy, shape=(5,))

    rho_out = Field.create_generic("rho", 2, index_shape=(1,))
    u_out = Field.create_generic("u", 2, index_shape=(2,))

    ispace = FullIterationSpace.create_from_slice(ctx, make_slice[:, :], f)
    ctx.set_iteration_space(ispace)

    asms = [
        Assignment(cx, (0, 1, -1, 0, 0)),
        Assignment(cy, (0, 0, 0, 1, -1)),
        Assignment(rho, sum(f.center(i) for i in range(5))),
        Assignment(u[0], 1 / rho * sum((f.center(i) * cxs[i]) for i in range(5))),
        Assignment(u[1], 1 / rho * sum((f.center(i) * cys[i]) for i in range(5))),
        Assignment(rho_out.center(0), rho),
        Assignment(u_out.center(0), u[0]),
        Assignment(u_out.center(1), u[1]),
    ]

    body = PsBlock(
        [PsComment("Compute and export density and velocity")]
        + [factory.parse_sympy(asm) for asm in asms]
    )

    ast = factory.loops_from_ispace(ispace, body)
    ast_clone = canon_clone(ast)

    for orig, clone in zip(dfs_preorder(ast), dfs_preorder(ast_clone), strict=True):
        assert type(orig) is type(clone)
        assert orig is not clone

        if isinstance(orig, PsSymbolExpr):
            assert isinstance(clone, PsSymbolExpr)

            if orig.symbol.name in ("ctr_0", "ctr_1", "rho", "u_0", "u_1", "cx", "cy"):
                assert clone.symbol.name == orig.symbol.name + "__0"
