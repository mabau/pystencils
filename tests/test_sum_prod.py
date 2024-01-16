# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pytest
import numpy as np

import pystencils.config
import sympy as sp
import sympy.abc

import pystencils as ps
from pystencils.typing import create_type


@pytest.mark.parametrize('dtype', ["float64", "float32"])
def test_sum(dtype):

    sum = sp.Sum(sp.abc.k, (sp.abc.k, 1, 100))
    expanded_sum = sum.doit()

    # print(sum)
    # print(expanded_sum)

    x = ps.fields(f'x: {dtype}[1d]')

    assignments = ps.AssignmentCollection({x.center(): sum})

    ast = ps.create_kernel(assignments)
    code = ps.get_code_str(ast)
    kernel = ast.compile()

    # ps.show_code(ast)

    if dtype == "float32":
        assert "5050.0f;" in code

    array = np.zeros((10,), dtype=dtype)
    kernel(x=array)
    assert np.allclose(array, int(expanded_sum) * np.ones_like(array))


@pytest.mark.parametrize('dtype', ["int32", "int64", "float64", "float32"])
def test_product(dtype):

    k = ps.TypedSymbol('k', create_type(dtype))

    sum = sympy.Product(k, (k, 1, 10))
    expanded_sum = sum.doit()

    # print(sum)
    # print(expanded_sum)

    x = ps.fields(f'x: {dtype}[1d]')

    assignments = ps.AssignmentCollection({x.center(): sum})

    config = pystencils.config.CreateKernelConfig()

    ast = ps.create_kernel(assignments, config=config)
    code = ps.get_code_str(ast)
    kernel = ast.compile()

    # print(code)
    if dtype == "int64" or dtype == "int32":
        assert '3628800;' in code
    elif dtype == "float32":
        assert '3628800.0f;' in code
    else:
        assert '3628800.0;' in code

    array = np.zeros((10,), dtype=dtype)
    kernel(x=array)
    assert np.allclose(array, int(expanded_sum) * np.ones_like(array))

# TODO: See Issue !55
# def test_prod_var_limit():
#
#     k = ps.TypedSymbol('k', create_type('int64'))
#     limit = ps.TypedSymbol('limit', create_type('int64'))
#
#     sum = sympy.Sum(k, (k, 1, limit))
#     expanded_sum = sum.replace(limit, 100).doit()
#
#     print(sum)
#     print(expanded_sum)
#
#     x = ps.fields('x: int64[1d]')
#
#     assignments = ps.AssignmentCollection({x.center(): sum})
#
#     ast = ps.create_kernel(assignments)
#     ps.show_code(ast)
#     kernel = ast.compile()
#
#     array = np.zeros((10,), np.int64)
#
#     kernel(x=array, limit=100)
#
#     assert np.allclose(array, int(expanded_sum) * np.ones_like(array))
