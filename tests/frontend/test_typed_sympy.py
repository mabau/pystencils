import numpy as np
import pickle
import sympy as sp
from sympy.logic import boolalg

from pystencils.sympyextensions.typed_sympy import (
    TypedSymbol,
    tcast,
    TypeAtom,
    DynamicType,
)
from pystencils.types import create_type
from pystencils.types.quick import UInt, Ptr


def test_type_atoms():
    atom1 = TypeAtom(create_type("int32"))
    atom2 = TypeAtom(create_type(np.int32))

    assert atom1 == atom2

    atom3 = TypeAtom(create_type("const int32"))
    assert atom1 != atom3

    atom4 = TypeAtom(DynamicType.INDEX_TYPE)
    atom5 = TypeAtom(DynamicType.NUMERIC_TYPE)

    assert atom3 != atom4
    assert atom4 != atom5

    dump = pickle.dumps(atom1)
    atom1_reconst = pickle.loads(dump)

    assert atom1_reconst == atom1


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


def test_casts():
    x, y = sp.symbols("x, y")

    #   Pickling
    expr = tcast(x, "int32")
    dump = pickle.dumps(expr)
    expr_reconst = pickle.loads(dump)
    assert expr_reconst == expr

    #   Boolean Casts
    bool_expr = tcast(x, "bool")
    assert isinstance(bool_expr, boolalg.Boolean)
    
    #   Check that we can construct boolean expressions with cast results
    _ = boolalg.Or(bool_expr, y)
    
    #   Assumptions
    expr = tcast(x, "int32")
    assert expr.is_integer
    assert expr.is_real
    assert expr.is_nonnegative is None

    expr = tcast(x, "uint32")
    assert expr.is_integer
    assert expr.is_real
    assert expr.is_nonnegative

    expr = tcast(x, "float32")
    assert expr.is_integer is None
    assert expr.is_real
    assert expr.is_nonnegative is None
