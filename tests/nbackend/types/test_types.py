import pytest
import numpy as np

from pystencils.backend.exceptions import PsInternalCompilerError
from pystencils.types import *
from pystencils.types.quick import *


@pytest.mark.parametrize(
    "Type", [PsSignedIntegerType, PsUnsignedIntegerType, PsIeeeFloatType]
)
def test_widths(Type):
    for width in Type.SUPPORTED_WIDTHS:
        assert Type(width).width == width

    for width in (1, 9, 33, 63):
        with pytest.raises(ValueError):
            Type(width)


def test_parsing_positive():
    assert make_type("const uint32_t * restrict") == Ptr(
        UInt(32, const=True), restrict=True
    )
    assert make_type("float * * const") == Ptr(Ptr(Fp(32)), const=True)
    assert make_type("uint16 * const") == Ptr(UInt(16), const=True)
    assert make_type("uint64 const * const") == Ptr(UInt(64, const=True), const=True)


def test_parsing_negative():
    bad_specs = [
        "const notatype * const",
        "cnost uint32_t",
        "uint45_t",
        "int",  # plain ints are ambiguous
        "float float",
        "double * int",
        "bool",
    ]

    for spec in bad_specs:
        with pytest.raises(ValueError):
            make_type(spec)


def test_numpy():
    import numpy as np

    assert make_type(np.single) == make_type(np.float32) == PsIeeeFloatType(32)
    assert (
        make_type(float)
        == make_type(np.double)
        == make_type(np.float64)
        == PsIeeeFloatType(64)
    )
    assert make_type(int) == make_type(np.int64) == PsSignedIntegerType(64)


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
    assert str(t) == "<anonymous>"
    with pytest.raises(PsTypeError):
        t.c_string()
