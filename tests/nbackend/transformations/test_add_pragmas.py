import sympy as sp
from itertools import product

from pystencils import make_slice, fields, Assignment
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
)

from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.structural import PsBlock, PsPragma, PsLoop
from pystencils.backend.transformations import InsertPragmasAtLoops, LoopPragma

def test_insert_pragmas():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    f, g = fields("f, g: [3D]")
    ispace = FullIterationSpace.create_from_slice(
        ctx, make_slice[:, :, :], archetype_field=f
    )
    ctx.set_iteration_space(ispace)

    stencil = list(product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
    loop_body = PsBlock([
        factory.parse_sympy(Assignment(f.center(0), sum(g.neighbors(stencil))))
    ])
    loops = factory.loops_from_ispace(ispace, loop_body)
    
    pragmas = (
        LoopPragma("omp parallel for", 0),
        LoopPragma("some nonsense pragma", 1),
        LoopPragma("omp simd", -1),
    )
    add_pragmas = InsertPragmasAtLoops(ctx, pragmas)
    ast = add_pragmas(loops)

    assert isinstance(ast, PsBlock)
    
    first_pragma = ast.statements[0]
    assert isinstance(first_pragma, PsPragma)
    assert first_pragma.text == pragmas[0].text

    assert ast.statements[1] == loops
    second_pragma = loops.body.statements[0]
    assert isinstance(second_pragma, PsPragma)
    assert second_pragma.text == pragmas[1].text

    second_loop = list(dfs_preorder(ast, lambda node: isinstance(node, PsLoop)))[1]
    assert isinstance(second_loop, PsLoop)
    third_pragma = second_loop.body.statements[0]
    assert isinstance(third_pragma, PsPragma)
    assert third_pragma.text == pragmas[2].text
