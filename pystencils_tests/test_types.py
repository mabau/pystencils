import sympy as sp
import numpy as np
import pystencils as ps
from pystencils import data_types
from pystencils.data_types import TypedSymbol, get_type_of_expression, VectorType, collate_types, create_type


def test_parsing():
    assert str(data_types.create_composite_type_from_string("const double *")) == "double const *"
    assert str(data_types.create_composite_type_from_string("double const *")) == "double const *"

    t1 = data_types.create_composite_type_from_string("const double * const * const restrict")
    t2 = data_types.create_composite_type_from_string(str(t1))
    assert t1 == t2


def test_collation():
    double_type = create_type("double")
    float_type = create_type("float32")
    double4_type = VectorType(double_type, 4)
    float4_type = VectorType(float_type, 4)
    assert collate_types([double_type, float_type]) == double_type
    assert collate_types([double4_type, float_type]) == double4_type
    assert collate_types([double4_type, float4_type]) == double4_type


def test_dtype_of_constants():
    # Some come constants are neither of type Integer,Float,Rational and don't have args
    # >>> isinstance(pi, Integer)
    # False
    # >>> isinstance(pi, Float)
    # False
    # >>> isinstance(pi, Rational)
    # False
    # >>> pi.args
    # ()
    get_type_of_expression(sp.pi)


def test_assumptions():
    x = ps.fields('x:  float32[3d]')

    assert x.shape[0].is_nonnegative
    assert (2 * x.shape[0]).is_nonnegative
    assert (2 * x.shape[0]).is_integer
    assert (TypedSymbol('a', create_type('uint64'))).is_nonnegative
    assert (TypedSymbol('a', create_type('uint64'))).is_positive is None
    assert (TypedSymbol('a', create_type('uint64')) + 1).is_positive
    assert (x.shape[0] + 1).is_real


def test_sqrt_of_integer():
    """Regression test for bug where sqrt(3) was classified as integer"""
    f = ps.fields("f: [1D]")
    tmp = sp.symbols("tmp")

    assignments = [ps.Assignment(tmp, sp.sqrt(3)),
                   ps.Assignment(f[0], tmp)]
    arr = np.array([1], dtype=np.float64)
    kernel = ps.create_kernel(assignments).compile()
    kernel(f=arr)
    assert 1.7 < arr[0] < 1.8
