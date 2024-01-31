import pytest
import numpy as np

from pystencils.nbackend.exceptions import PsInternalCompilerError
from pystencils.nbackend.types import *
from pystencils.nbackend.types.quick import *


@pytest.mark.parametrize(
    "numpy_type",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    ],
)
def test_numpy_translation(numpy_type):
    dtype_obj = np.dtype(numpy_type)
    ps_type = make_type(numpy_type)

    assert isinstance(ps_type, PsNumericType)
    assert ps_type.numpy_dtype == dtype_obj
    assert ps_type.itemsize == dtype_obj.itemsize

    assert isinstance(ps_type.create_constant(13), numpy_type)

    if ps_type.is_int():
        with pytest.raises(PsTypeError):
            ps_type.create_constant(13.0)
        with pytest.raises(PsTypeError):
            ps_type.create_constant(1.75)

    if ps_type.is_sint():
        assert numpy_type(17) == ps_type.create_constant(17)
        assert numpy_type(-4) == ps_type.create_constant(-4)

    if ps_type.is_uint():
        with pytest.raises(PsTypeError):
            ps_type.create_constant(-4)

    if ps_type.is_float():
        assert numpy_type(17.3) == ps_type.create_constant(17.3)
        assert numpy_type(-4.2) == ps_type.create_constant(-4.2)


def test_constify():
    t = PsCustomType("std::shared_ptr< Custom >")
    assert deconstify(t) == t
    assert deconstify(constify(t)) == t
    s = PsCustomType("Field", const=True)
    assert constify(s) == s


def test_struct_types():
    t = PsStructType(
        [
            PsStructType.Member("data", Ptr(Fp(32))),
            ("size", UInt(32)),
        ]
    )

    assert t.anonymous
    with pytest.raises(PsInternalCompilerError):
        str(t)
