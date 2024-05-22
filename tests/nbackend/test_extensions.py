
import sympy as sp

from pystencils import make_slice, Field, Assignment
from pystencils.backend.kernelcreation import KernelCreationContext, AstFactory, FullIterationSpace
from pystencils.backend.transformations import CanonicalizeSymbols, HoistLoopInvariantDeclarations
from pystencils.backend.literals import PsLiteral
from pystencils.backend.emission import CAstPrinter
from pystencils.backend.ast.expressions import PsExpression, PsSubscript
from pystencils.backend.ast.structural import PsBlock, PsDeclaration
from pystencils.types.quick import Arr, Int


def test_literals():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    f = Field.create_generic("f", 3)
    x = sp.Symbol("x")
    
    cells = PsExpression.make(PsLiteral("CELLS", Arr(Int(64, const=True), 3)))
    global_constant = PsExpression.make(PsLiteral("C", ctx.default_dtype))

    loop_slice = make_slice[
        0:PsSubscript(cells, factory.parse_index(0)),
        0:PsSubscript(cells, factory.parse_index(1)),
        0:PsSubscript(cells, factory.parse_index(2)),
    ]

    ispace = FullIterationSpace.create_from_slice(ctx, loop_slice)
    ctx.set_iteration_space(ispace)
    
    x_decl = PsDeclaration(factory.parse_sympy(x), global_constant)

    loop_body = PsBlock([
        x_decl,
        factory.parse_sympy(Assignment(f.center(), x))
    ])

    loops = factory.loops_from_ispace(ispace, loop_body)
    ast = PsBlock([loops])
    
    canon = CanonicalizeSymbols(ctx)
    ast = canon(ast)

    hoist = HoistLoopInvariantDeclarations(ctx)
    ast = hoist(ast)

    assert isinstance(ast, PsBlock)
    assert len(ast.statements) == 2
    assert ast.statements[0] == x_decl

    code = CAstPrinter()(ast)
    print(code)

    assert "const double x = C;" in code
    assert "CELLS[((int64_t) 0)]" in code
    assert "CELLS[((int64_t) 1)]" in code
    assert "CELLS[((int64_t) 2)]" in code
