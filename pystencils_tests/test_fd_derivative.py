import sympy as sp
from pystencils import fields
from pystencils.fd import Diff, diff, collect_diffs
from pystencils.fd.derivative import replace_generic_laplacian


def test_fs():
    f = sp.Symbol("f", commutative=False)

    a = Diff(Diff(Diff(f, 1), 0), 0)
    assert a.is_commutative is False
    print(str(a))

    assert diff(f) == f

    x, y = sp.symbols("x, y")
    collected_terms = collect_diffs(diff(x, 0, 0))
    assert collected_terms == Diff(Diff(x, 0, -1), 0, -1)

    src = fields("src : double[2D]")
    expr = sp.Add(Diff(Diff(src[0, 0])), 10)
    expected = Diff(Diff(src[0, 0], 0, -1), 0, -1) + Diff(Diff(src[0, 0], 1, -1), 1, -1) + 10
    result = replace_generic_laplacian(expr, 3)
    assert result == expected