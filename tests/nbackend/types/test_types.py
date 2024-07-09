import pytest
import numpy as np
import pickle

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
    assert create_type("const uint32_t * restrict") is Ptr(
        UInt(32, const=True), restrict=True
    )
    assert create_type("float * * const") is Ptr(
        Ptr(Fp(32), restrict=False), const=True, restrict=False
    )
    assert create_type("float * * restrict const") is Ptr(
        Ptr(Fp(32), restrict=False), const=True, restrict=True
    )
    assert create_type("uint16 * const") is Ptr(UInt(16), const=True, restrict=False)
    assert create_type("uint64 const * const") is Ptr(
        UInt(64, const=True), const=True, restrict=False
    )


def test_parsing_negative():
    bad_specs = [
        "const notatype * const",
        "cnost uint32_t",
        "uint45_t",
        "float float",
        "double * int",
    ]

    for spec in bad_specs:
        with pytest.raises(ValueError):
            create_type(spec)


def test_numpy():
    import numpy as np

    assert create_type(np.single) is create_type(np.float32) is PsIeeeFloatType(32)
    assert (
        create_type(float)
        is create_type(np.double)
        is create_type(np.float64)
        is PsIeeeFloatType(64)
    )
    assert create_type(int) is create_type(np.int64) is PsSignedIntegerType(64)


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
    ps_type = create_type(numpy_type)

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
    assert deconstify(t) is t
    assert deconstify(constify(t)) is t

    s = PsCustomType("Field", const=True)
    assert constify(s) is s

    i32 = create_type(np.int32)
    i32_2 = PsSignedIntegerType(32)

    assert i32 is i32_2
    assert constify(i32) is constify(i32_2)

    i32_const = PsSignedIntegerType(32, const=True)
    assert i32_const is not i32
    assert i32_const is constify(i32)


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

    t = PsStructType([
        ("a", SInt(8)),
        ("b", SInt(16)),
        ("c", SInt(64))
    ])

    #   Check that natural alignment is taken into account
    numpy_type = np.dtype([("a", "i1"), ("b", "i2"), ("c", "i8")], align=True)
    assert t.numpy_dtype == numpy_type
    assert t.itemsize == numpy_type.itemsize == 16


def test_pickle():
    types = [
        Bool(const=True),
        Bool(const=False),
        Custom("std::vector< uint_t >", const=False),
        Ptr(Fp(32, const=False), restrict=True, const=True),
        SInt(32, const=True),
        SInt(16, const=False),
        UInt(8, const=False),
        UInt(width=16, const=False),
        Int(width=32, const=False),
        Fp(width=16, const=True),
        PsStructType([("x", UInt(32)), ("y", UInt(32)), ("val", Fp(64))], "myStruct"),
        PsStructType([("data", Fp(32))], "None"),
        PsArrayType(Fp(16, const=True), 42),
        PsArrayType(PsVectorType(Fp(32), 8, const=False), 42)
    ]

    dumped = pickle.dumps(types)
    restored = pickle.loads(dumped)

    for t1, t2 in zip(types, restored):
        assert t1 == t2
