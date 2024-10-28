import pytest

from pystencils import create_type
from pystencils.backend.kernelcreation import KernelCreationContext, AstFactory, Typifier
from pystencils.backend.memory import PsSymbol, BufferBasePtr
from pystencils.backend.constants import PsConstant
from pystencils.backend.ast.expressions import (
    PsExpression,
    PsCast,
    PsMemAcc,
    PsArrayInitList,
    PsSubscript,
    PsBufferAcc,
    PsSymbolExpr,
)
from pystencils.backend.ast.structural import (
    PsStatement,
    PsAssignment,
    PsDeclaration,
    PsBlock,
    PsConditional,
    PsComment,
    PsPragma,
    PsLoop,
)
from pystencils.types.quick import Fp, Ptr


def test_cloning():
    ctx = KernelCreationContext()
    typify = Typifier(ctx)

    x, y, z, m = [PsExpression.make(ctx.get_symbol(name)) for name in "xyzm"]
    q = PsExpression.make(ctx.get_symbol("q", create_type("bool")))
    a, b, c = [PsExpression.make(ctx.get_symbol(name, ctx.index_dtype)) for name in "abc"]
    c1 = PsExpression.make(PsConstant(3.0))
    c2 = PsExpression.make(PsConstant(-1.0))
    one_f = PsExpression.make(PsConstant(1.0))
    one_i = PsExpression.make(PsConstant(1))

    def check(orig, clone):
        assert not (orig is clone)
        assert type(orig) is type(clone)
        assert orig.structurally_equal(clone)
        
        if isinstance(orig, PsExpression):
            #   Regression: Expression data types used to not be cloned
            assert orig.dtype == clone.dtype

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
            q, PsBlock([PsStatement(x + y)]), PsBlock([PsComment("hello world")])
        ),
        PsDeclaration(
            m,
            PsArrayInitList([
                [x, y, one_f + x],
                [one_f, c2, z]
            ])
        ),
        PsPragma("omp parallel for"),
        PsLoop(
            a,
            b,
            c,
            one_i,
            PsBlock(
                [
                    PsComment("Loop body"),
                    PsAssignment(x, y),
                    PsAssignment(x, y),
                    PsPragma("#pragma clang loop vectorize(enable)"),
                    PsStatement(
                        PsMemAcc(PsCast(Ptr(Fp(32)), z), one_i)
                        + PsCast(Fp(32), PsSubscript(m, (one_i + one_i + one_i, b + one_i)))
                    ),
                ]
            ),
        ),
    ]:
        ast = typify(ast)
        ast_clone = ast.clone()
        check(ast, ast_clone)


def test_buffer_acc():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    from pystencils import fields

    f, g = fields("f, g(3): [2D]")
    a, b = [ctx.get_symbol(n, ctx.index_dtype) for n in "ab"]

    f_buf = ctx.get_buffer(f)

    f_acc = PsBufferAcc(f_buf.base_pointer, [PsExpression.make(i) for i in (a, b)] + [factory.parse_index(0)])
    assert f_acc.buffer == f_buf
    assert f_acc.base_pointer.structurally_equal(PsSymbolExpr(f_buf.base_pointer))

    f_acc_clone = f_acc.clone()
    assert f_acc_clone is not f_acc

    assert f_acc_clone.buffer == f_buf
    assert f_acc_clone.base_pointer.structurally_equal(PsSymbolExpr(f_buf.base_pointer))
    assert len(f_acc_clone.index) == 3
    assert f_acc_clone.index[0].structurally_equal(PsSymbolExpr(ctx.get_symbol("a")))
    assert f_acc_clone.index[1].structurally_equal(PsSymbolExpr(ctx.get_symbol("b")))

    g_buf = ctx.get_buffer(g)

    g_acc = PsBufferAcc(g_buf.base_pointer, [PsExpression.make(i) for i in (a, b)] + [factory.parse_index(2)])
    assert g_acc.buffer == g_buf
    assert g_acc.base_pointer.structurally_equal(PsSymbolExpr(g_buf.base_pointer))

    second_bptr = PsExpression.make(ctx.get_symbol("data_g_interior", g_buf.base_pointer.dtype))
    second_bptr.symbol.add_property(BufferBasePtr(g_buf))
    g_acc.base_pointer = second_bptr

    assert g_acc.base_pointer == second_bptr
    assert g_acc.buffer == g_buf

    #   cannot change base pointer to different buffer
    with pytest.raises(ValueError):
        g_acc.base_pointer = PsExpression.make(f_buf.base_pointer)
