import numpy as np

from pystencils.sympyextensions.typed_sympy import (
    TypedSymbol,
    CastFunc,
    TypeAtom,
    DynamicType,
)
from pystencils.types import create_type
from pystencils.types.quick import UInt, Ptr


def test_type_atoms():
    atom1 = TypeAtom(create_type("int32"))
    atom2 = TypeAtom(create_type("int32"))

    assert atom1 == atom2

    atom3 = TypeAtom(create_type("const int32"))
    assert atom1 != atom3

    atom4 = TypeAtom(DynamicType.INDEX_TYPE)
    atom5 = TypeAtom(DynamicType.NUMERIC_TYPE)

    assert atom3 != atom4
    assert atom4 != atom5


def test_typed_symbol():
    x = TypedSymbol("x", "uint32")
    x2 = TypedSymbol("x", "uint64 *")
    z = TypedSymbol("z", "float32")

    assert x == TypedSymbol("x", np.uint32)
    assert x != x2

    assert x.dtype == UInt(32)
    assert x2.dtype == Ptr(UInt(64))

    assert x.is_integer
    assert x.is_nonnegative

    assert not x2.is_integer

    assert z.is_real
    assert not z.is_nonnegative


def test_cast_func():
    assert (
        CastFunc(TypedSymbol("s", np.uint), np.int64).canonical
        == TypedSymbol("s", np.uint).canonical
    )

    a = CastFunc(5, np.uint)
    assert a.is_negative is False
    assert a.is_nonnegative
