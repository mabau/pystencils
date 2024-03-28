import pytest
import sympy as sp
import numpy as np

from pystencils import Assignment, TypedSymbol, Field, FieldType

from pystencils.backend.ast.structural import PsDeclaration
from pystencils.backend.ast.expressions import PsConstantExpr, PsSymbolExpr, PsBinOp
from pystencils.types import constify
from pystencils.types.quick import Fp, create_numeric_type
from pystencils.backend.kernelcreation.context import KernelCreationContext
from pystencils.backend.kernelcreation.freeze import FreezeExpressions
from pystencils.backend.kernelcreation.typification import Typifier, TypificationError

from pystencils.sympyextensions.integer_functions import (
    bit_shift_left,
    bit_shift_right,
    bitwise_and,
    bitwise_xor,
    bitwise_or,
)


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
    ctx = KernelCreationContext(default_dtype=create_numeric_type(np.float64))
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


def test_typify_integer_binops():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    ctx.get_symbol("x", ctx.index_dtype)
    ctx.get_symbol("y", ctx.index_dtype)

    x, y = sp.symbols("x, y")
    expr = bit_shift_left(
        bit_shift_right(bitwise_and(2, 2), bitwise_or(x, y)), bitwise_xor(2, 2)
    )
    expr = freeze(expr)
    expr = typify(expr)

    def check(expr):
        match expr:
            case PsConstantExpr(cs):
                assert cs.value == 2
                assert cs.dtype == constify(ctx.index_dtype)
            case PsSymbolExpr(symb):
                assert symb.name in "xyz"
                assert symb.dtype == ctx.index_dtype
            case PsBinOp(op1, op2):
                check(op1)
                check(op2)
            case _:
                pytest.fail(f"Unexpected expression: {expr}")

    check(expr)


def test_typify_integer_binops_floating_arg():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x = sp.Symbol("x")
    expr = bit_shift_left(x, 2)
    expr = freeze(expr)

    with pytest.raises(TypificationError):
        expr = typify(expr)


def test_typify_integer_binops_in_floating_context():
    ctx = KernelCreationContext()
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    ctx.get_symbol("i", ctx.index_dtype)

    x, i = sp.symbols("x, i")
    expr = x + bit_shift_left(i, 2)
    expr = freeze(expr)

    with pytest.raises(TypificationError):
        expr = typify(expr)


def test_regression_typify_constants():
    ctx = KernelCreationContext(default_dtype=Fp(32))
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x, y = sp.symbols("x, y")
    expr = (-x - y) ** 2

    typify(freeze(expr))  # just test that no error is raised
