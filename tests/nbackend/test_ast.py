from pystencils.backend.symbols import PsSymbol
from pystencils.backend.constants import PsConstant
from pystencils.backend.ast.expressions import (
    PsExpression,
    PsCast,
    PsDeref,
    PsSubscript,
)
from pystencils.backend.ast.structural import (
    PsStatement,
    PsAssignment,
    PsBlock,
    PsConditional,
    PsComment,
    PsPragma,
    PsLoop,
)
from pystencils.types.quick import Fp, Ptr


def test_cloning():
    x, y, z = [PsExpression.make(PsSymbol(name)) for name in "xyz"]
    c1 = PsExpression.make(PsConstant(3.0))
    c2 = PsExpression.make(PsConstant(-1.0))
    one = PsExpression.make(PsConstant(1))

    def check(orig, clone):
        assert not (orig is clone)
        assert type(orig) is type(clone)
        assert orig.structurally_equal(clone)

        for c1, c2 in zip(orig.children, clone.children, strict=True):
            check(c1, c2)

    for ast in [
        x,
        y,
        c1,
        x + y,
        x / y + c1,
        c1 + c2,
        PsStatement(x * y * z + c1),
        PsAssignment(y, x / c1),
        PsBlock([PsAssignment(x, c1 * y), PsAssignment(z, c2 + c1 * z)]),
        PsConditional(
            y, PsBlock([PsStatement(x + y)]), PsBlock([PsComment("hello world")])
        ),
        PsPragma("omp parallel for"),
        PsLoop(
            x,
            y,
            z,
            one,
            PsBlock(
                [
                    PsComment("Loop body"),
                    PsAssignment(x, y),
                    PsAssignment(x, y),
                    PsPragma("#pragma clang loop vectorize(enable)"),
                    PsStatement(
                        PsDeref(PsCast(Ptr(Fp(32)), z))
                        + PsSubscript(z, one + one + one)
                    ),
                ]
            ),
        ),
    ]:
        ast_clone = ast.clone()
        check(ast, ast_clone)
