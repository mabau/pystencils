import sympy as sp
import pystencils as ps
from pystencils.fast_approximation import insert_fast_divisions, insert_fast_sqrts, fast_sqrt, fast_inv_sqrt, \
    fast_division


def test_fast_sqrt():
    f, g = ps.fields("f, g: double[2D]")
    expr = sp.sqrt(f[0, 0] + f[1, 0])

    assert len(insert_fast_sqrts(expr).atoms(fast_sqrt)) == 1
    assert len(insert_fast_sqrts([expr])[0].atoms(fast_sqrt)) == 1

    expr = 3 / sp.sqrt(f[0, 0] + f[1, 0])
    assert len(insert_fast_sqrts(expr).atoms(fast_inv_sqrt)) == 1
    ac = ps.AssignmentCollection([expr], [])
    assert len(insert_fast_sqrts(ac).main_assignments[0].atoms(fast_inv_sqrt)) == 1


def test_fast_divisions():
    f, g = ps.fields("f, g: double[2D]")
    expr = f[0, 0] / f[1, 0]
    assert len(insert_fast_divisions(expr).atoms(fast_division)) == 1

    expr = 1 / f[0, 0] * 2 / f[0, 1]
    assert len(insert_fast_divisions(expr).atoms(fast_division)) == 1

    ast = ps.create_kernel(ps.Assignment(g[0, 0], insert_fast_divisions(expr)), target='gpu')
    code_str = str(ps.show_code(ast))
    assert '__fdividef' in code_str
