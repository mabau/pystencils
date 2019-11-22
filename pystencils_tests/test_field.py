import numpy as np
import pytest
import sympy as sp

import pystencils as ps
from pystencils.field import Field, FieldType, layout_string_to_tuple


def test_field_basic():
    f = Field.create_generic('f', spatial_dimensions=2)
    assert FieldType.is_generic(f)
    assert f['E'] == f[1, 0]
    assert f['N'] == f[0, 1]
    assert '_' in f.center._latex('dummy')

    f = Field.create_fixed_size('f', (10, 10), strides=(80, 8), dtype=np.float64)
    assert f.spatial_strides == (10, 1)
    assert f.index_strides == ()
    assert f.center_vector == sp.Matrix([f.center])

    f = Field.create_fixed_size('f', (8, 8, 2, 2), index_dimensions=2)
    assert f.center_vector == sp.Matrix([[f(0, 0), f(0, 1)],
                                         [f(1, 0), f(1, 1)]])
    field_access = f[1, 1]
    assert field_access.nr_of_coordinates == 2
    assert field_access.offset_name == 'NE'
    neighbor = field_access.neighbor(coord_id=0, offset=-2)
    assert neighbor.offsets == (-1, 1)
    assert '_' in neighbor._latex('dummy')

    f = Field.create_generic('f', spatial_dimensions=5, index_dimensions=2)
    field_access = f[1, -1, 2, -3, 0](1, 0)
    assert field_access.offsets == (1, -1, 2, -3, 0)
    assert field_access.index == (1, 0)


def test_error_handling():
    struct_dtype = np.dtype([('a', np.int32), ('b', np.float64), ('c', np.uint32)])
    Field.create_generic('f', spatial_dimensions=2, index_dimensions=0, dtype=struct_dtype)
    with pytest.raises(ValueError) as e:
        Field.create_generic('f', spatial_dimensions=2, index_dimensions=1, dtype=struct_dtype)
    assert 'index dimension' in str(e.value)

    arr = np.array([[1, 2.0, 3], [1, 2.0, 3]], dtype=struct_dtype)
    Field.create_from_numpy_array('f', arr, index_dimensions=0)
    with pytest.raises(ValueError) as e:
        Field.create_from_numpy_array('f', arr, index_dimensions=1)
    assert 'Structured arrays' in str(e.value)

    arr = np.zeros([3, 3, 3])
    Field.create_from_numpy_array('f', arr, index_dimensions=2)
    with pytest.raises(ValueError) as e:
        Field.create_from_numpy_array('f', arr, index_dimensions=3)
    assert 'Too many' in str(e.value)

    Field.create_fixed_size('f', (3, 2, 4), index_dimensions=0, dtype=struct_dtype, layout='reverse_numpy')
    with pytest.raises(ValueError) as e:
        Field.create_fixed_size('f', (3, 2, 4), index_dimensions=1, dtype=struct_dtype, layout='reverse_numpy')
    assert 'Structured arrays' in str(e.value)

    f = Field.create_fixed_size('f', (10, 10))
    with pytest.raises(ValueError) as e:
        f[1]
    assert 'Wrong number of spatial indices' in str(e.value)

    f = Field.create_generic('f', spatial_dimensions=2, index_shape=(3,))
    with pytest.raises(ValueError) as e:
        f(3)
    assert 'out of bounds' in str(e.value)

    f = Field.create_fixed_size('f', (10, 10, 3, 4), index_dimensions=2)
    with pytest.raises(ValueError) as e:
        f(3, 0)
    assert 'out of bounds' in str(e.value)

    with pytest.raises(ValueError) as e:
        f(1, 0)(1, 0)
    assert 'Indexing an already indexed' in str(e.value)

    with pytest.raises(ValueError) as e:
        f(1)
    assert 'Wrong number of indices' in str(e.value)

    with pytest.raises(ValueError) as e:
        Field.create_generic('f', spatial_dimensions=2, layout='wrong')
    assert 'Unknown layout descriptor' in str(e.value)

    assert layout_string_to_tuple('fzyx', dim=4) == (3, 2, 1, 0)
    with pytest.raises(ValueError) as e:
        layout_string_to_tuple('wrong', dim=4)
    assert 'Unknown layout descriptor' in str(e.value)


def test_decorator_scoping():
    dst = ps.fields('dst : double[2D]')

    def f1():
        a = sp.Symbol("a")

        def f2():
            b = sp.Symbol("b")

            @ps.kernel
            def decorated_func():
                dst[0, 0] @= a + b

            return decorated_func

        return f2

    assert f1()(), ps.Assignment(dst[0, 0], sp.Symbol("a") + sp.Symbol("b"))


def test_string_creation():
    x, y, z = ps.fields('  x(4),    y(3,5) z : double[  3,  47]')
    assert x.index_shape == (4,)
    assert y.index_shape == (3, 5)
    assert z.spatial_shape == (3, 47)


def test_itemsize():

    x = ps.fields('x: float32[1d]')
    y = ps.fields('y:  float64[2d]')
    i = ps.fields('i:  int16[1d]')

    assert x.itemsize == 4
    assert y.itemsize == 8
    assert i.itemsize == 2


def test_staggered():

    # D2Q5
    j1, j2, j3 = ps.fields('j1(2), j2(2,2), j3(2,2,2) : double[2D]', field_type=FieldType.STAGGERED)

    assert j1[0, 1](1) == j1.staggered_access((0, sp.Rational(1, 2)))
    assert j1[1, 1](1) == j1.staggered_access((1, sp.Rational(1, 2)))
    assert j1[0, 2](1) == j1.staggered_access((0, sp.Rational(3, 2)))
    assert j1[0, 1](1) == j1.staggered_access("N")
    assert j1[0, 0](1) == j1.staggered_access("S")

    assert j2[0, 1](1, 1) == j2.staggered_access((0, sp.Rational(1, 2)), 1)
    assert j2[0, 1](1, 1) == j2.staggered_access("N", 1)

    assert j3[0, 1](1, 1, 1) == j3.staggered_access((0, sp.Rational(1, 2)), (1, 1))
    assert j3[0, 1](1, 1, 1) == j3.staggered_access("N", (1, 1))

    # D2Q9
    k = ps.fields('k(4) : double[2D]', field_type=FieldType.STAGGERED)

    assert k[1, 1](2) == k.staggered_access("NE")
    assert k[0, 0](2) == k.staggered_access("SW")

    assert k[0, 0](3) == k.staggered_access("NW")

    # sign reversed when using as flux field
    r = ps.fields('r(2) : double[2D]', field_type=FieldType.STAGGERED_FLUX)
    assert r[0, 0](0) == r.staggered_access("W")
    assert -r[1, 0](0) == r.staggered_access("E")
