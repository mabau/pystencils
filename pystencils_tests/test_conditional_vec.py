import numpy as np
import sympy as sp
import pytest

import pystencils as ps
from pystencils.astnodes import Block, Conditional
from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets
from pystencils.cpu.vectorization import vec_all, vec_any


@pytest.mark.skipif(not get_supported_instruction_sets(), reason='cannot detect CPU instruction set')
@pytest.mark.skipif('neon' in get_supported_instruction_sets(), reason='ARM does not have collective instructions')
def test_vec_any():
    data_arr = np.zeros((15, 15))

    data_arr[3:9, 1] = 1.0
    data = ps.fields("data: double[2D]", data=data_arr)

    c = [
        ps.Assignment(sp.Symbol("t1"), vec_any(data.center() > 0.0)),
        Conditional(vec_any(data.center() > 0.0), Block([
            ps.Assignment(data.center(), 2.0)
        ]))
    ]
    instruction_set = get_supported_instruction_sets()[-1]
    ast = ps.create_kernel(c, target='cpu',
                           cpu_vectorize_info={'instruction_set': instruction_set})
    kernel = ast.compile()
    kernel(data=data_arr)

    width = ast.instruction_set['width']

    np.testing.assert_equal(data_arr[3:9, 0:width], 2.0)


@pytest.mark.skipif(not get_supported_instruction_sets(), reason='cannot detect CPU instruction set')
@pytest.mark.skipif('neon' in get_supported_instruction_sets(), reason='ARM does not have collective instructions')
def test_vec_all():
    data_arr = np.zeros((15, 15))

    data_arr[3:9, 2:7] = 1.0
    data = ps.fields("data: double[2D]", data=data_arr)

    c = [
        Conditional(vec_all(data.center() > 0.0), Block([
            ps.Assignment(data.center(), 2.0)
        ]))
    ]
    ast = ps.create_kernel(c, target='cpu',
                           cpu_vectorize_info={'instruction_set': get_supported_instruction_sets()[-1]})
    kernel = ast.compile()
    before = data_arr.copy()
    kernel(data=data_arr)
    np.testing.assert_equal(data_arr, before)


@pytest.mark.skipif(not get_supported_instruction_sets(), reason='cannot detect CPU instruction set')
def test_boolean_before_loop():
    t1, t2 = sp.symbols('t1, t2')
    f_arr = np.ones((10, 10))
    g_arr = np.zeros_like(f_arr)
    f, g = ps.fields("f, g : double[2D]", f=f_arr, g=g_arr)

    a = [
        ps.Assignment(t1, t2 > 0),
        ps.Assignment(g[0, 0],
                      sp.Piecewise((f[0, 0], t1), (42, True)))
    ]
    ast = ps.create_kernel(a, cpu_vectorize_info={'instruction_set': get_supported_instruction_sets()[-1]})
    kernel = ast.compile()
    kernel(f=f_arr, g=g_arr, t2=1.0)
    print(g)
    np.testing.assert_array_equal(g_arr, 1.0)
    kernel(f=f_arr, g=g_arr, t2=-1.0)
    np.testing.assert_array_equal(g_arr, 42.0)
