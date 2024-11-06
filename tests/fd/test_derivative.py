import sympy as sp
from sympy.abc import a, b, t, x, y, z
from sympy.printing.latex import LatexPrinter

import pystencils as ps
from pystencils.fd import *


def test_derivative_basic():
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
    assert ps.fd.Diff(0) == 0

    expr = ps.fd.diff(ps.fd.diff(x, 0, 0), 1, 1)
    assert expr.get_arg_recursive() == x
    assert expr.change_arg_recursive(y).get_arg_recursive() == y


def test_derivative_expand_collect():
    original = Diff(x*y*z)
    result = combine_diff_products(combine_diff_products(expand_diff_products(original))).expand()
    assert original == result

    original = -3 * y * z * Diff(x) + 2 * x * z * Diff(y)
    result = expand_diff_products(combine_diff_products(original)).expand()
    assert original == result

    original = a + b * Diff(x ** 2 * y * z)
    expanded = expand_diff_products(original)
    collect_res = combine_diff_products(combine_diff_products(combine_diff_products(expanded)))
    assert collect_res == original


def test_diff_expand_using_linearity():
    eps = sp.symbols("epsilon")
    funcs = [a, b]
    test = Diff(eps * Diff(a+b))
    result = expand_diff_linear(test, functions=funcs)
    assert result == eps * Diff(Diff(a)) + eps * Diff(Diff(b))
