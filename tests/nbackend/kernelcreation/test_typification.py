import pytest
import sympy as sp
import numpy as np
import pymbolic.primitives as pb

from pystencils import Assignment, TypedSymbol, Field, FieldType

from pystencils.nbackend.ast import PsDeclaration
from pystencils.nbackend.types import constify, deconstify, PsStructType
from pystencils.nbackend.types.quick import *
from pystencils.nbackend.typed_expressions import PsTypedConstant, PsTypedVariable
from pystencils.nbackend.kernelcreation.options import KernelCreationOptions
from pystencils.nbackend.kernelcreation.context import KernelCreationContext
from pystencils.nbackend.kernelcreation.freeze import FreezeExpressions
from pystencils.nbackend.kernelcreation.typification import Typifier, TypificationError


def test_typify_simple():
    options = KernelCreationOptions()
    ctx = KernelCreationContext(options)
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x, y, z = sp.symbols("x, y, z")
    asm = Assignment(z, 2 * x + y)

    fasm = freeze(asm)
    fasm = typify(fasm)

    assert isinstance(fasm, PsDeclaration)

    def check(expr):
        match expr:
            case PsTypedConstant(value, dtype):
                assert value == 2
                assert dtype == constify(ctx.options.default_dtype)
            case PsTypedVariable(name, dtype):
                assert name in "xyz"
                assert dtype == ctx.options.default_dtype
            case pb.Sum(cs) | pb.Product(cs):
                [check(c) for c in cs]
            case _:
                pytest.fail(f"Unexpected expression: {expr}")

    check(fasm.lhs.expression)
    check(fasm.rhs.expression)


def test_typify_structs():
    options = KernelCreationOptions(default_dtype=Fp(32))
    ctx = KernelCreationContext(options)
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
    options = KernelCreationOptions()
    ctx = KernelCreationContext(options)
    freeze = FreezeExpressions(ctx)
    typify = Typifier(ctx)

    x, y, z = sp.symbols("x, y, z")
    expr = freeze(2 * x + 3 * y + z - 4)
    expr = typify(expr)

    def check(expr):
        match expr:
            case PsTypedConstant(value, dtype):
                assert value in (2, 3, -4)
                assert dtype == constify(ctx.options.default_dtype)
            case PsTypedVariable(name, dtype):
                assert name in "xyz"
                assert dtype == ctx.options.default_dtype
            case pb.Sum(cs) | pb.Product(cs):
                [check(c) for c in cs]
            case _:
                pytest.fail(f"Unexpected expression: {expr}")

    check(expr.expression)


def test_erronous_typing():
    options = KernelCreationOptions(default_dtype=make_numeric_type(np.float64))
    ctx = KernelCreationContext(options)
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
