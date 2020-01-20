# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import numpy as np
import sympy
from sympy.abc import k

import pystencils
from pystencils.data_types import create_type


def test_sum():

    sum = sympy.Sum(k, (k, 1, 100))
    expanded_sum = sum.doit()

    print(sum)
    print(expanded_sum)

    x = pystencils.fields('x: float32[1d]')

    assignments = pystencils.AssignmentCollection({
        x.center(): sum
    })

    ast = pystencils.create_kernel(assignments)
    code = str(pystencils.get_code_obj(ast))
    kernel = ast.compile()

    print(code)
    assert 'double sum' in code

    array = np.zeros((10,), np.float32)

    kernel(x=array)

    assert np.allclose(array, int(expanded_sum) * np.ones_like(array))


def test_sum_use_float():

    sum = sympy.Sum(k, (k, 1, 100))
    expanded_sum = sum.doit()

    print(sum)
    print(expanded_sum)

    x = pystencils.fields('x: float32[1d]')

    assignments = pystencils.AssignmentCollection({
        x.center(): sum
    })

    ast = pystencils.create_kernel(assignments, data_type=create_type('float32'))
    code = str(pystencils.get_code_obj(ast))
    kernel = ast.compile()

    print(code)
    print(pystencils.get_code_obj(ast))
    assert 'float sum' in code

    array = np.zeros((10,), np.float32)

    kernel(x=array)

    assert np.allclose(array, int(expanded_sum) * np.ones_like(array))


def test_product():

    k = pystencils.TypedSymbol('k', create_type('int64'))

    sum = sympy.Product(k, (k, 1, 10))
    expanded_sum = sum.doit()

    print(sum)
    print(expanded_sum)

    x = pystencils.fields('x: int64[1d]')

    assignments = pystencils.AssignmentCollection({
        x.center(): sum
    })

    ast = pystencils.create_kernel(assignments)
    code = pystencils.get_code_str(ast)
    kernel = ast.compile()

    print(code)
    assert 'int64_t product' in code

    array = np.zeros((10,), np.int64)

    kernel(x=array)

    assert np.allclose(array, int(expanded_sum) * np.ones_like(array))


def test_prod_var_limit():

    k = pystencils.TypedSymbol('k', create_type('int64'))
    limit = pystencils.TypedSymbol('limit', create_type('int64'))

    sum = sympy.Sum(k, (k, 1, limit))
    expanded_sum = sum.replace(limit, 100).doit()

    print(sum)
    print(expanded_sum)

    x = pystencils.fields('x: int64[1d]')

    assignments = pystencils.AssignmentCollection({
        x.center(): sum
    })

    ast = pystencils.create_kernel(assignments)
    pystencils.show_code(ast)
    kernel = ast.compile()

    array = np.zeros((10,), np.int64)

    kernel(x=array, limit=100)

    assert np.allclose(array, int(expanded_sum) * np.ones_like(array))
