# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.
"""

"""

import itertools

import numpy as np
import pytest
import sympy
from sympy.functions import im, re

import pystencils
from pystencils import AssignmentCollection
from pystencils.data_types import TypedSymbol, create_type

X, Y = pystencils.fields('x, y: complex64[2d]')
A, B = pystencils.fields('a, b: float32[2d]')
S1, S2, T = sympy.symbols('S1, S2, T')

TEST_ASSIGNMENTS = [
    AssignmentCollection({X[0, 0]: 1j}),
    AssignmentCollection({
        S1: re(Y.center),
        S2: im(Y.center),
        X[0, 0]: 2j * S1 + S2
    }),
    AssignmentCollection({
        A.center: re(Y.center),
        B.center: im(Y.center),
    }),
    AssignmentCollection({
        Y.center: re(Y.center) + X.center + 2j,
    }),
    AssignmentCollection({
        T: 2 + 4j,
        Y.center: X.center / T,
    })
]

SCALAR_DTYPES = ['float32', 'float64']


@pytest.mark.parametrize("assignment, scalar_dtypes",
                         itertools.product(TEST_ASSIGNMENTS, (np.float32,)))
@pytest.mark.parametrize('target', ('cpu', 'gpu'))
def test_complex_numbers(assignment, scalar_dtypes, target):
    ast = pystencils.create_kernel(assignment,
                                   target=target,
                                   data_type=scalar_dtypes)
    code = str(pystencils.show_code(ast))

    print(code)
    assert "Not supported" not in code

    kernel = ast.compile()
    assert kernel is not None


X, Y = pystencils.fields('x, y: complex128[2d]')
A, B = pystencils.fields('a, b: float64[2d]')
S1, S2 = sympy.symbols('S1, S2')
T128 = TypedSymbol('ts', create_type('complex128'))

TEST_ASSIGNMENTS = [
    AssignmentCollection({X[0, 0]: 1j}),
    AssignmentCollection({
        S1: re(Y.center),
        S2: im(Y.center),
        X[0, 0]: 2j * S1 + S2
    }),
    AssignmentCollection({
        A.center: re(Y.center),
        B.center: im(Y.center),
    }),
    AssignmentCollection({
        Y.center: re(Y.center) + X.center + 2j,
    }),
    AssignmentCollection({
        T128: 2 + 4j,
        Y.center: X.center / T128,
    })
]

SCALAR_DTYPES = ['float64']


@pytest.mark.parametrize("assignment", TEST_ASSIGNMENTS)
@pytest.mark.parametrize('target', ('cpu', 'gpu'))
def test_complex_numbers_64(assignment, target):
    ast = pystencils.create_kernel(assignment,
                                   target=target,
                                   data_type='double')
    code = str(pystencils.show_code(ast))

    print(code)
    assert "Not supported" not in code

    kernel = ast.compile()
    assert kernel is not None


@pytest.mark.parametrize('dtype', (np.float32, np.float64))
@pytest.mark.parametrize('target', ('cpu', 'gpu'))
def test_complex_execution(dtype, target):

    complex_dtype = f'complex{64 if dtype ==np.float32 else 128}'
    x, y = pystencils.fields(f'x, y:  {complex_dtype}[2d]')

    x_arr = np.zeros((20, 30), complex_dtype)
    y_arr = np.zeros((20, 30), complex_dtype)

    assignments = AssignmentCollection({
        y.center: x.center * (2j+1)
    })

    if target == 'gpu':
        from pycuda.gpuarray import zeros
        x_arr = zeros((20, 30), complex_dtype)
        y_arr = zeros((20, 30), complex_dtype)

    kernel = pystencils.create_kernel(assignments, target=target, data_type=dtype).compile()

    kernel(x=x_arr, y=y_arr)
