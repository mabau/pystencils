import sympy as sp
from sympy.printing.latex import LatexPrinter
from pystencils.fd import *


def test_derivative_basic():
    x, y, z, t = sp.symbols("x y z t")
    d = diff

    op1, op2, op3 = DiffOperator(), DiffOperator(target=x), DiffOperator(target=x, superscript=1)
    d1, d2, d3 = Diff(t), Diff(t, target=x), Diff(t, target=x, superscript=1)
    printer = LatexPrinter()
    assert all('\\partial' in l._latex(printer) for l in (op1, op2, op3, d1, d2, d3))

    dx, dy = DiffOperator(target=x), DiffOperator(target=y)
    diff_term = (dx + dy) ** 2 + 1
    diff_term = diff_term.expand()
    assert diff_term == dx**2 + 2 * dx * dy + dy**2 + 1

    assert DiffOperator.apply(diff_term, t) == d(t, x, x) + 2 * d(t, x, y) + d(t, y, y) + t

