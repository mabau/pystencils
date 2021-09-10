import pytest
import sympy as sp

import pystencils as ps
from pystencils.fast_approximation import (
    fast_division, fast_inv_sqrt, fast_sqrt, insert_fast_divisions, insert_fast_sqrts)


def test_fast_sqrt():
    pytest.importorskip('pycuda')
    f, g = ps.fields("f, g: double[2D]")
    expr = sp.sqrt(f[0, 0] + f[1, 0])

    assert len(insert_fast_sqrts(expr).atoms(fast_sqrt)) == 1
    assert len(insert_fast_sqrts([expr])[0].atoms(fast_sqrt)) == 1
    ast_gpu = ps.create_kernel(ps.Assignment(g[0, 0], insert_fast_sqrts(expr)), target=ps.Target.GPU)
    ast_gpu.compile()
    code_str = ps.get_code_str(ast_gpu)
    assert '__fsqrt_rn' in code_str

    expr = ps.Assignment(sp.Symbol("tmp"), 3 / sp.sqrt(f[0, 0] + f[1, 0]))
    assert len(insert_fast_sqrts(expr).atoms(fast_inv_sqrt)) == 1

    ac = ps.AssignmentCollection([expr], [])
    assert len(insert_fast_sqrts(ac).main_assignments[0].atoms(fast_inv_sqrt)) == 1
    ast_gpu = ps.create_kernel(insert_fast_sqrts(ac), target=ps.Target.GPU)
    ast_gpu.compile()
    code_str = ps.get_code_str(ast_gpu)
    assert '__frsqrt_rn' in code_str


def test_fast_divisions():
    pytest.importorskip('pycuda')
    f, g = ps.fields("f, g: double[2D]")
    expr = f[0, 0] / f[1, 0]
    assert len(insert_fast_divisions(expr).atoms(fast_division)) == 1

    expr = 1 / f[0, 0] * 2 / f[0, 1]
    assert len(insert_fast_divisions(expr).atoms(fast_division)) == 1

    ast = ps.create_kernel(ps.Assignment(g[0, 0], insert_fast_divisions(expr)), target=ps.Target.GPU)
    ast.compile()
    code_str = ps.get_code_str(ast)
    assert '__fdividef' in code_str
