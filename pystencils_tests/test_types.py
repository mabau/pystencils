import pytest

import pystencils.config
import sympy as sp
import numpy as np

import pystencils as ps
from pystencils.typing import TypedSymbol, get_type_of_expression, VectorType, collate_types, create_type, \
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


# TODO redo following tests
def test_collation():
    double_type = BasicType('float64')
    float_type = BasicType('float32')
    double4_type = VectorType(double_type, 4)
    float4_type = VectorType(float_type, 4)
    assert collate_types([double_type, float_type]) == double_type
    assert collate_types([double4_type, float_type]) == double4_type
    assert collate_types([double4_type, float4_type]) == double4_type


def test_vector_type():
    double_type = BasicType("double")
    float_type = BasicType('float32')
    double4_type = VectorType(double_type, 4)
    float4_type = VectorType(float_type, 4)

    assert double4_type.item_size == 4
    assert float4_type.item_size == 4

    assert not double4_type == 4


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


def test_sqrt_of_integer():
    """Regression test for bug where sqrt(3) was classified as integer"""
    f = ps.fields("f: [1D]")
    tmp = sp.symbols("tmp")

    assignments = [ps.Assignment(tmp, sp.sqrt(3)),
                   ps.Assignment(f[0], tmp)]
    arr_double = np.array([1], dtype=np.float64)
    kernel = ps.create_kernel(assignments).compile()
    kernel(f=arr_double)
    assert 1.7 < arr_double[0] < 1.8

    f = ps.fields("f: float32[1D]")
    tmp = sp.symbols("tmp")

    assignments = [ps.Assignment(tmp, sp.sqrt(3)),
                   ps.Assignment(f[0], tmp)]
    arr_single = np.array([1], dtype=np.float32)
    config = pystencils.config.CreateKernelConfig(data_type="float32")
    kernel = ps.create_kernel(assignments, config=config).compile()
    kernel(f=arr_single)

    code = ps.get_code_str(kernel.ast)

    assert "1.7320508075688772f" in code
    assert 1.7 < arr_single[0] < 1.8


def test_integer_comparision():
    f = ps.fields("f [2D]")
    d = sp.Symbol("dir")

    ur = ps.Assignment(f[0, 0], sp.Piecewise((0, sp.Equality(d, 1)), (f[0, 0], True)))

    ast = ps.create_kernel(ur)
    code = ps.get_code_str(ast)

    assert "_data_f_00[_stride_f_1*ctr_1] = ((((dir) == (1))) ? (0.0): (_data_f_00[_stride_f_1*ctr_1]));" in code


def test_Basic_data_type():
    assert typed_symbols(("s", "f"), np.uint) == typed_symbols("s, f", np.uint)
    t_symbols = typed_symbols(("s", "f"), np.uint)
    s = t_symbols[0]

    assert t_symbols[0] == TypedSymbol("s", np.uint)
    assert s.dtype.is_uint()
    assert s.dtype.is_complex() == 0

    assert typed_symbols("s", str).dtype.is_other()
    assert typed_symbols("s", bool).dtype.is_other()
    assert typed_symbols("s", np.void).dtype.is_other()

    assert typed_symbols("s", np.float64).dtype.c_name == 'double'
    # removed for old sympy version
    # assert typed_symbols(("s"), np.float64).dtype.sympy_dtype == typed_symbols(("s"), float).dtype.sympy_dtype

    f, g = ps.fields("f, g : double[2D]")

    expr = ps.Assignment(f.center(), 2 * g.center() + 5)
    new_expr = type_all_numbers(expr, np.float64)

    assert "cast_func(2, double)" in str(new_expr)
    assert "cast_func(5, double)" in str(new_expr)

    m = matrix_symbols("a, b", np.uint, 3, 3)
    assert len(m) == 2
    m = m[0]
    for i, elem in enumerate(m):
        assert elem == TypedSymbol(f"a{i}", np.uint)
        assert elem.dtype.is_uint()

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

    up = [ps.Assignment(tau, 1.0 / (0.5 + (3.0 * m))),
          ps.Assignment(f.center, tau)]

    ast = ps.create_kernel(up, config=pystencils.config.CreateKernelConfig(data_type="float32"))
    code = ps.get_code_str(ast)

    assert "1.0f" in code


def test_pow():
    f = ps.fields('f(10): float32[2D]')
    m, tau = sp.symbols("m, tau")

    up = [ps.Assignment(tau, m ** 1.5),
          ps.Assignment(f.center, tau)]

    ast = ps.create_kernel(up, config=pystencils.config.CreateKernelConfig(data_type="float32"))
    code = ps.get_code_str(ast)

    assert "1.5f" in code
