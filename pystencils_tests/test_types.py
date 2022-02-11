import pytest

import pystencils.config
import sympy as sp
import numpy as np

import pystencils as ps
from pystencils.typing import TypedSymbol, get_type_of_expression, VectorType, collate_types, \
    typed_symbols, CastFunc, PointerArithmeticFunc, PointerType, result_type, BasicType


def test_result_type():
    i = np.dtype('int32')
    l = np.dtype('int64')
    ui = np.dtype('uint32')
    ul = np.dtype('uint64')
    f = np.dtype('float32')
    d = np.dtype('float64')
    b = np.dtype('bool')

    assert result_type(i, l) == l
    assert result_type(l, i) == l
    assert result_type(ui, i) == i
    assert result_type(ui, l) == l
    assert result_type(ul, i) == i
    assert result_type(ul, l) == l
    assert result_type(d, f) == d
    assert result_type(f, d) == d
    assert result_type(i, f) == f
    assert result_type(l, f) == f
    assert result_type(ui, f) == f
    assert result_type(ul, f) == f
    assert result_type(i, d) == d
    assert result_type(l, d) == d
    assert result_type(ui, d) == d
    assert result_type(ul, d) == d
    assert result_type(b, i) == i
    assert result_type(b, l) == l
    assert result_type(b, ui) == ui
    assert result_type(b, ul) == ul
    assert result_type(b, f) == f
    assert result_type(b, d) == d


@pytest.mark.parametrize('dtype', ('float64', 'float32', 'int64', 'int32', 'uint32', 'uint64'))
def test_simple_add(dtype):
    constant = 1.0
    if dtype[0] in 'ui':
        constant = 1
    f = ps.fields(f"f: {dtype}[1D]")
    d = TypedSymbol("d", dtype)

    test_arr = np.array([constant], dtype=dtype)

    ur = ps.Assignment(f[0], f[0] + d)

    ast = ps.create_kernel(ur)
    code = ps.get_code_str(ast)
    kernel = ast.compile()
    kernel(f=test_arr, d=constant)

    assert test_arr[0] == constant+constant


@pytest.mark.parametrize('dtype1', ('float64', 'float32', 'int64', 'int32', 'uint32', 'uint64'))
@pytest.mark.parametrize('dtype2', ('float64', 'float32', 'int64', 'int32', 'uint32', 'uint64'))
def test_mixed_add(dtype1, dtype2):

    constant = 1
    f = ps.fields(f"f: {dtype1}[1D]")
    g = ps.fields(f"g: {dtype2}[1D]")

    test_f = np.array([constant], dtype=dtype1)
    test_g = np.array([constant], dtype=dtype2)

    ur = ps.Assignment(f[0], f[0] + g[0])

    # TODO Markus: check for the logging if colate_types(dtype1, dtype2) != dtype1
    ast = ps.create_kernel(ur)
    code = ps.get_code_str(ast)
    kernel = ast.compile()
    kernel(f=test_f, g=test_g)

    assert test_f[0] == constant+constant


def test_collation():
    double_type = BasicType('float64')
    float_type = BasicType('float32')
    double4_type = VectorType(double_type, 4)
    float4_type = VectorType(float_type, 4)
    assert collate_types([double_type, float_type]) == double_type
    assert collate_types([double4_type, float_type]) == double4_type
    assert collate_types([double4_type, float4_type]) == double4_type


def test_vector_type():
    double_type = BasicType('float64')
    float_type = BasicType('float32')
    double4_type = VectorType(double_type, 4)
    float4_type = VectorType(float_type, 4)

    assert double4_type.item_size == 4
    assert float4_type.item_size == 4

    double4_type2 = VectorType(double_type, 4)
    assert double4_type == double4_type2
    assert double4_type != 4
    assert double4_type != float4_type


def test_pointer_type():
    double_type = BasicType('float64')
    float_type = BasicType('float32')
    double4_type = PointerType(double_type, restrict=True)
    float4_type = PointerType(float_type, restrict=False)

    assert double4_type.item_size == 1
    assert float4_type.item_size == 1

    assert not double4_type == 4

    assert not double4_type.alias
    assert float4_type.alias


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
    assert (TypedSymbol('a', BasicType('uint64'))).is_nonnegative
    assert (TypedSymbol('a', BasicType('uint64'))).is_positive is None
    assert (TypedSymbol('a', BasicType('uint64')) + 1).is_positive
    assert (x.shape[0] + 1).is_real


@pytest.mark.parametrize('dtype', ('float64', 'float32'))
def test_sqrt_of_integer(dtype):
    """Regression test for bug where sqrt(3) was classified as integer"""
    f = ps.fields(f'f: {dtype}[1D]')
    tmp = sp.symbols('tmp')

    assignments = [ps.Assignment(tmp, sp.sqrt(3)),
                   ps.Assignment(f[0], tmp)]
    arr = np.array([1], dtype=dtype)
    config = pystencils.config.CreateKernelConfig(data_type=dtype, default_number_float=dtype)

    ast = ps.create_kernel(assignments, config=config)
    kernel = ast.compile()
    kernel(f=arr)
    assert 1.7 < arr[0] < 1.8

    code = ps.get_code_str(ast)
    constant = '1.7320508075688772f'
    if dtype == 'float32':
        assert constant in code
    else:
        assert constant not in code


@pytest.mark.parametrize('dtype', ('float64', 'float32'))
def test_integer_comparision(dtype):
    f = ps.fields(f"f: {dtype}[2D]")
    d = TypedSymbol("dir", "int64")

    ur = ps.Assignment(f[0, 0], sp.Piecewise((0, sp.Equality(d, 1)), (f[0, 0], True)))

    ast = ps.create_kernel(ur)
    code = ps.get_code_str(ast)

    # There should be an explicit cast for the integer zero to the type of the field on the rhs
    if dtype == 'float64':
        t = "_data_f_00[_stride_f_1*ctr_1] = ((((dir) == (1))) ? (0.0): (_data_f_00[_stride_f_1*ctr_1]));"
    else:
        t = "_data_f_00[_stride_f_1*ctr_1] = ((((dir) == (1))) ? (0.0f): (_data_f_00[_stride_f_1*ctr_1]));"
    assert t in code


def test_typed_symbols_dtype():
    assert typed_symbols(("s", "f"), np.uint) == typed_symbols("s, f", np.uint)
    t_symbols = typed_symbols(("s", "f"), np.uint)
    s = t_symbols[0]

    assert t_symbols[0] == TypedSymbol("s", np.uint)
    assert s.dtype.is_uint()

    assert typed_symbols("s", np.float64).dtype.c_name == 'double'
    assert typed_symbols("s", np.float32).dtype.c_name == 'float'

    assert TypedSymbol("s", np.uint).canonical == TypedSymbol("s", np.uint)
    assert TypedSymbol("s", np.uint).reversed == TypedSymbol("s", np.uint)


def test_cast_func():
    assert CastFunc(TypedSymbol("s", np.uint), np.int64).canonical == TypedSymbol("s", np.uint).canonical

    a = CastFunc(5, np.uint)
    assert a.is_negative is False
    assert a.is_nonnegative


def test_pointer_arithmetic_func():
    assert PointerArithmeticFunc(TypedSymbol("s", np.uint), 1).canonical == TypedSymbol("s", np.uint).canonical


def test_division():
    f = ps.fields('f(10): float32[2D]')
    m, tau = sp.symbols("m, tau")

    up = [ps.Assignment(tau, 1 / (0.5 + (3.0 * m))),
          ps.Assignment(f.center, tau)]
    config = pystencils.config.CreateKernelConfig(data_type='float32', default_number_float='float32')
    ast = ps.create_kernel(up, config=config)
    code = ps.get_code_str(ast)

    assert "((1.0f) / (m*3.0f + 0.5f))" in code


def test_pow():
    f = ps.fields('f(10): float32[2D]')
    m, tau = sp.symbols("m, tau")

    up = [ps.Assignment(tau, m ** 1.5),
          ps.Assignment(f.center, tau)]

    config = pystencils.config.CreateKernelConfig(data_type="float32", default_number_float='float32')
    ast = ps.create_kernel(up, config=config)
    code = ps.get_code_str(ast)

    assert "1.5f" in code
