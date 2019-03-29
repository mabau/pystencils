import pystencils as ps
import sympy as sp
import numpy as np
from pystencils.astnodes import Conditional, Block
from pystencils.cpu.vectorization import vec_all, vec_any


def test_vec_any():
    data_arr = np.zeros((15, 15))

    data_arr[3:9, 2:7] = 1.0
    data = ps.fields("data: double[2D]", data=data_arr)

    c = [
        ps.Assignment(sp.Symbol("t1"), vec_any(data.center() > 0.0)),
        Conditional(vec_any(data.center() > 0.0), Block([
            ps.Assignment(data.center(), 2.0)
        ]))
    ]
    ast = ps.create_kernel(c, target='cpu',
                           cpu_vectorize_info={'instruction_set': 'avx'})
    kernel = ast.compile()
    kernel(data=data_arr)
    np.testing.assert_equal(data_arr[3:9, 0:8], 2.0)


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
                           cpu_vectorize_info={'instruction_set': 'avx'})
    kernel = ast.compile()
    before = data_arr.copy()
    kernel(data=data_arr)
    np.testing.assert_equal(data_arr, before)
