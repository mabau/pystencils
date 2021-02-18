import pytest

import numpy as np
import sympy as sp

import pystencils as ps
from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets

supported_instruction_sets = get_supported_instruction_sets() if get_supported_instruction_sets() else []


@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
def test_vectorisation_varying_arch(instruction_set):
    shape = (9, 9, 3)
    arr = np.ones(shape, order='f')

    @ps.kernel
    def update_rule(s):
        f = ps.fields("f(3) : [2D]", f=arr)
        s.tmp0 @= f(0)
        s.tmp1 @= f(1)
        s.tmp2 @= f(2)
        f0, f1, f2 = f(0), f(1), f(2)
        f0 @= 2 * s.tmp0
        f1 @= 2 * s.tmp0
        f2 @= 2 * s.tmp0

    ast = ps.create_kernel(update_rule, cpu_vectorize_info={'instruction_set': instruction_set})
    kernel = ast.compile()
    kernel(f=arr)
    np.testing.assert_equal(arr, 2)


@pytest.mark.parametrize('dtype', ('float', 'double'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
def test_vectorized_abs(instruction_set, dtype):
    """Some instructions sets have abs, some don't.
       Furthermore, the special treatment of unary minus makes this data type-sensitive too.
    """
    arr = np.ones((2 ** 2 + 2, 2 ** 3 + 2), dtype=np.float64 if dtype == 'double' else np.float32)
    arr[-3:, :] = -1

    f, g = ps.fields(f=arr, g=arr)
    update_rule = [ps.Assignment(g.center(), sp.Abs(f.center()))]

    ast = ps.create_kernel(update_rule, cpu_vectorize_info={'instruction_set': instruction_set})

    func = ast.compile()
    dst = np.zeros_like(arr)
    func(g=dst, f=arr)
    np.testing.assert_equal(np.sum(dst[1:-1, 1:-1]), 2 ** 2 * 2 ** 3)
