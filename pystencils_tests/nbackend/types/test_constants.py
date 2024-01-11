import pytest

from pystencils.nbackend.types.quick import *
from pystencils.nbackend.types import PsTypeError
from pystencils.nbackend.typed_expressions import PsTypedConstant


@pytest.mark.parametrize("width", (8, 16, 32, 64))
def test_integer_constants(width):
    dtype = SInt(width)
    a = PsTypedConstant(42, dtype)
    b = PsTypedConstant(2, dtype)

    assert a + b == PsTypedConstant(44, dtype)
    assert a - b == PsTypedConstant(40, dtype)
    assert a * b == PsTypedConstant(84, dtype)

    assert a - b != PsTypedConstant(-12, dtype)

    #   Typed constants only compare to themselves
    assert a + b != 44


@pytest.mark.parametrize("width", (32, 64))
def test_float_constants(width):
    a = PsTypedConstant(32.0, Fp(width))
    b = PsTypedConstant(0.5, Fp(width))
    c = PsTypedConstant(2.0, Fp(width))

    assert a + b == PsTypedConstant(32.5, Fp(width))
    assert a * b == PsTypedConstant(16.0, Fp(width))
    assert a - b == PsTypedConstant(31.5, Fp(width))
    assert a / c == PsTypedConstant(16.0, Fp(width))

        
def test_illegal_ops():
    #   Cannot interpret negative numbers as unsigned types
    with pytest.raises(PsTypeError):
        _ = PsTypedConstant(-3, UInt(32))

    #   Mixed ops are illegal
    with pytest.raises(PsTypeError):
        _ = PsTypedConstant(32.0, Fp(32)) + PsTypedConstant(2, UInt(32))

    with pytest.raises(PsTypeError):
        _ = PsTypedConstant(32.0, Fp(32)) - PsTypedConstant(2, UInt(32))

    with pytest.raises(PsTypeError):
        _ = PsTypedConstant(32.0, Fp(32)) * PsTypedConstant(2, UInt(32))

    with pytest.raises(PsTypeError):
        _ = PsTypedConstant(32.0, Fp(32)) / PsTypedConstant(2, UInt(32))


@pytest.mark.parametrize("width", (8, 16, 32, 64))
def test_integer_division(width):
    a = PsTypedConstant(-5, SInt(width))
    b = PsTypedConstant(2, SInt(width))

    assert a / b == PsTypedConstant(-2, SInt(width))
    assert a % b == PsTypedConstant(-1, SInt(width))
