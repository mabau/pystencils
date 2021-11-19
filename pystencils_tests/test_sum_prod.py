# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import pytest
import numpy as np
import sympy as sp
import sympy.abc

import pystencils as ps
from pystencils.data_types import create_type


@pytest.mark.parametrize('default_assignment_simplifications', [False, True])
def test_sum(default_assignment_simplifications):

    sum = sp.Sum(sp.abc.k, (sp.abc.k, 1, 100))
    expanded_sum = sum.doit()

    print(sum)
    print(expanded_sum)

    x = ps.fields('x: float32[1d]')

    assignments = ps.AssignmentCollection({x.center(): sum})

    config = ps.CreateKernelConfig(default_assignment_simplifications=default_assignment_simplifications)
    ast = ps.create_kernel(assignments, config=config)
    code = ps.get_code_str(ast)
    kernel = ast.compile()

    print(code)
    if default_assignment_simplifications is False:
        assert 'double sum' in code

    array = np.zeros((10,), np.float32)

    kernel(x=array)

    assert np.allclose(array, int(expanded_sum) * np.ones_like(array))


@pytest.mark.parametrize('default_assignment_simplifications', [False, True])
def test_sum_use_float(default_assignment_simplifications):

    sum = sympy.Sum(sp.abc.k, (sp.abc.k, 1, 100))
    expanded_sum = sum.doit()

    print(sum)
    print(expanded_sum)

    x = ps.fields('x: float32[1d]')

    assignments = ps.AssignmentCollection({x.center(): sum})

    config = ps.CreateKernelConfig(default_assignment_simplifications=default_assignment_simplifications,
                                   data_type=create_type('float32'))
    ast = ps.create_kernel(assignments, config=config)
    code = ps.get_code_str(ast)
    kernel = ast.compile()

    print(code)
    if default_assignment_simplifications is False:
        assert 'float sum' in code

    array = np.zeros((10,), np.float32)

    kernel(x=array)

    assert np.allclose(array, int(expanded_sum) * np.ones_like(array))


@pytest.mark.parametrize('default_assignment_simplifications', [False, True])
def test_product(default_assignment_simplifications):

    k = ps.TypedSymbol('k', create_type('int64'))

    sum = sympy.Product(k, (k, 1, 10))
    expanded_sum = sum.doit()

    print(sum)
    print(expanded_sum)

    x = ps.fields('x: int64[1d]')

    assignments = ps.AssignmentCollection({x.center(): sum})

    config = ps.CreateKernelConfig(default_assignment_simplifications=default_assignment_simplifications)

    ast = ps.create_kernel(assignments, config=config)
    code = ps.get_code_str(ast)
    kernel = ast.compile()

    print(code)
    if default_assignment_simplifications is False:
        assert 'int64_t product' in code

    array = np.zeros((10,), np.int64)

    kernel(x=array)

    assert np.allclose(array, int(expanded_sum) * np.ones_like(array))


def test_prod_var_limit():

    k = ps.TypedSymbol('k', create_type('int64'))
    limit = ps.TypedSymbol('limit', create_type('int64'))

    sum = sympy.Sum(k, (k, 1, limit))
    expanded_sum = sum.replace(limit, 100).doit()

    print(sum)
    print(expanded_sum)

    x = ps.fields('x: int64[1d]')

    assignments = ps.AssignmentCollection({x.center(): sum})

    ast = ps.create_kernel(assignments)
    ps.show_code(ast)
    kernel = ast.compile()

    array = np.zeros((10,), np.int64)

    kernel(x=array, limit=100)

    assert np.allclose(array, int(expanded_sum) * np.ones_like(array))
