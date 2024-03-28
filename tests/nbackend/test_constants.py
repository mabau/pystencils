import numpy as np
import pytest

from pystencils.types import PsTypeError
from pystencils.backend.constants import PsConstant
from pystencils.types.quick import Fp, Bool, UInt, SInt
from pystencils.backend.exceptions import PsInternalCompilerError


def test_constant_equality():
    c1 = PsConstant(1.0, Fp(32))
    c2 = PsConstant(1.0, Fp(32))

    assert c1 == c2
    assert hash(c1) == hash(c2)

    c3 = PsConstant(1.0, Fp(64))
    assert c1 != c3
    assert hash(c1) != hash(c3)

    c4 = c1.reinterpret_as(Fp(64))
    assert c4 != c1
    assert c4 == c3


def test_interpret():
    c1 = PsConstant(3.4, Fp(32))
    c2 = PsConstant(3.4)

    assert c2.interpret_as(Fp(32)) == c1

    with pytest.raises(PsInternalCompilerError):
        _ = c1.interpret_as(Fp(64))


def test_boolean_constants():
    true = PsConstant(True, Bool())
    for val in (1, 1.0, True, np.True_):
        assert PsConstant(val, Bool()) == true

    false = PsConstant(False, Bool())
    for val in (0, 0.0, False, np.False_):
        assert PsConstant(val, Bool()) == false

    with pytest.raises(PsTypeError):
        PsConstant(1.1, Bool())


def test_integer_bounds():
    #  should not throw:
    for val in (255, np.uint8(255), np.int16(255), np.int64(255)):
        _ = PsConstant(val, UInt(8))

    for val in (-128, np.int16(-128), np.int64(-128)):
        _ = PsConstant(val, SInt(8))
    
    #  should throw:
    for val in (256, np.int16(256), np.int64(256)):
        with pytest.raises(PsTypeError):
            _ = PsConstant(val, UInt(8))

    for val in (-42, np.int32(-42)):
        with pytest.raises(PsTypeError):
            _ = PsConstant(val, UInt(8))

    for val in (-129, np.int16(-129), np.int64(-129)):
        with pytest.raises(PsTypeError):
            _ = PsConstant(val, SInt(8))


def test_floating_bounds():
    for val in (5.1e4, -5.9e4):
        _ = PsConstant(val, Fp(16))
        _ = PsConstant(val, Fp(32))
        _ = PsConstant(val, Fp(64))

    for val in (8.1e5, -7.6e5):
        with pytest.raises(PsTypeError):
            _ = PsConstant(val, Fp(16))
