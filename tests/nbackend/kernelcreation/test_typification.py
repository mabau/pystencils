import pytest
import sympy as sp
import numpy as np

from pystencils import Assignment, TypedSymbol, Field, FieldType

from pystencils.backend.ast.structural import PsDeclaration
from pystencils.backend.ast.expressions import PsConstantExpr, PsSymbolExpr, PsBinOp
from pystencils.backend.types import constify
from pystencils.backend.types.quick import Fp, make_numeric_type
from pystencils.backend.kernelcreation.context import KernelCreationContext
from pystencils.backend.kernelcreation.freeze import FreezeExpressions
from pystencils.backend.kernelcreation.typification import Typifier, TypificationError


def test_typify_simple():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x, y, z = sp.symbols("x, y, z")
    asm = Assignment(z, 2 * x + y)

    fasm = freeze(asm)
    fasm = typify(fasm)

    assert isinstance(fasm, PsDeclaration)

    def check(expr):
        match expr:
            case PsConstantExpr(cs):
                assert cs.value == 2
                assert cs.dtype == constify(ctx.default_dtype)
            case PsSymbolExpr(symb):
                assert symb.name in "xyz"
                assert symb.dtype == ctx.default_dtype
            case PsBinOp(op1, op2):
                check(op1)
                check(op2)
            case _:
                pytest.fail(f"Unexpected expression: {expr}")

    check(fasm.lhs)
    check(fasm.rhs)


def test_typify_structs():
    ctx = KernelCreationContext(default_dtype=Fp(32))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    np_struct = np.dtype([("size", np.uint32), ("data", np.float32)])
    f = Field.create_generic("f", 1, dtype=np_struct, field_type=FieldType.CUSTOM)
    x = sp.Symbol("x")

    #   Good
    asm = Assignment(x, f.absolute_access((0,), "data"))
    fasm = freeze(asm)
    fasm = typify(fasm)

    #   Bad
    asm = Assignment(x, f.absolute_access((0,), "size"))
    fasm = freeze(asm)
    with pytest.raises(TypificationError):
        fasm = typify(fasm)


def test_contextual_typing():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x, y, z = sp.symbols("x, y, z")
    expr = freeze(2 * x + 3 * y + z - 4)
    expr = typify(expr)

    def check(expr):
        match expr:
            case PsConstantExpr(cs):
                assert cs.value in (2, 3, -4)
                assert cs.dtype == constify(ctx.default_dtype)
            case PsSymbolExpr(symb):
                assert symb.name in "xyz"
                assert symb.dtype == ctx.default_dtype
            case PsBinOp(op1, op2):
                check(op1)
                check(op2)
            case _:
                pytest.fail(f"Unexpected expression: {expr}")

    check(expr)


def test_erronous_typing():
    ctx = KernelCreationContext(default_dtype=make_numeric_type(np.float64))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x, y, z = sp.symbols("x, y, z")
    q = TypedSymbol("q", np.float32)
    w = TypedSymbol("w", np.float16)

    expr = freeze(2 * x + 3 * y + q - 4)

    with pytest.raises(TypificationError):
        typify(expr)

    asm = Assignment(q, 3 - w)
    fasm = freeze(asm)
    with pytest.raises(TypificationError):
        typify(fasm)

    asm = Assignment(q, 3 - x)
    fasm = freeze(asm)
    with pytest.raises(TypificationError):
        typify(fasm)
