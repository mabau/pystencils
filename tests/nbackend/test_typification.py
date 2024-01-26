import pytest
import sympy as sp
import pymbolic.primitives as pb

from pystencils import Assignment

from pystencils.nbackend.ast import PsDeclaration
from pystencils.nbackend.types import constify
from pystencils.nbackend.typed_expressions import PsTypedConstant, PsTypedVariable
from pystencils.nbackend.kernelcreation.options import KernelCreationOptions
from pystencils.nbackend.kernelcreation.context import KernelCreationContext
from pystencils.nbackend.kernelcreation.freeze import FreezeExpressions
from pystencils.nbackend.kernelcreation.typification import Typifier


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
            case pb.Variable:
                pytest.fail("Encountered untyped variable")
            case pb.Sum(cs) | pb.Product(cs):
                [check(c) for c in cs]
            case _:
                pytest.fail("Non-exhaustive pattern matcher.")

    check(fasm.lhs.expression)
    check(fasm.rhs.expression)
