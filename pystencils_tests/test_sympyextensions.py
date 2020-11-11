import sympy
import pystencils

from pystencils.sympyextensions import replace_second_order_products
from pystencils.sympyextensions import remove_higher_order_terms
from pystencils.sympyextensions import complete_the_squares_in_exp
from pystencils.sympyextensions import extract_most_common_factor
from pystencils.sympyextensions import count_operations
from pystencils.sympyextensions import common_denominator
from pystencils.sympyextensions import get_symmetric_part

from pystencils import Assignment
from pystencils.fast_approximation import (fast_division, fast_inv_sqrt, fast_sqrt,
                                           insert_fast_divisions, insert_fast_sqrts)


def test_replace_second_order_products():
    x, y = sympy.symbols('x y')
    expr = 4 * x * y
    expected_expr_positive = 2 * ((x + y) ** 2 - x ** 2 - y ** 2)
    expected_expr_negative = 2 * (-(x - y) ** 2 + x ** 2 + y ** 2)

    result = replace_second_order_products(expr, search_symbols=[x, y], positive=True)
    assert result == expected_expr_positive
    assert (result - expected_expr_positive).simplify() == 0

    result = replace_second_order_products(expr, search_symbols=[x, y], positive=False)
    assert result == expected_expr_negative
    assert (result - expected_expr_negative).simplify() == 0

    result = replace_second_order_products(expr, search_symbols=[x, y], positive=None)
    assert result == expected_expr_positive

    a = [Assignment(sympy.symbols('z'), x + y)]
    replace_second_order_products(expr, search_symbols=[x, y], positive=True, replace_mixed=a)
    assert len(a) == 2


def test_remove_higher_order_terms():
    x, y = sympy.symbols('x y')

    expr = sympy.Mul(x, y)

    result = remove_higher_order_terms(expr, order=1, symbols=[x, y])
    assert result == 0
    result = remove_higher_order_terms(expr, order=2, symbols=[x, y])
    assert result == expr

    expr = sympy.Pow(x, 3)

    result = remove_higher_order_terms(expr, order=2, symbols=[x, y])
    assert result == 0
    result = remove_higher_order_terms(expr, order=3, symbols=[x, y])
    assert result == expr


def test_complete_the_squares_in_exp():
    a, b, c, s, n = sympy.symbols('a b c s n')
    expr = a * s ** 2 + b * s + c
    result = complete_the_squares_in_exp(expr, symbols_to_complete=[s])
    assert result == expr

    expr = sympy.exp(a * s ** 2 + b * s + c)
    expected_result = sympy.exp(a*s**2 + c - b**2 / (4*a))
    result = complete_the_squares_in_exp(expr, symbols_to_complete=[s])
    assert result == expected_result


def test_extract_most_common_factor():
    x, y = sympy.symbols('x y')
    expr = 1 / (x + y) + 3 / (x + y) + 3 / (x + y)
    most_common_factor = extract_most_common_factor(expr)

    assert most_common_factor[0] == 7
    assert sympy.prod(most_common_factor) == expr

    expr = 1 / x + 3 / (x + y) + 3 / y
    most_common_factor = extract_most_common_factor(expr)

    assert most_common_factor[0] == 3
    assert sympy.prod(most_common_factor) == expr

    expr = 1 / x
    most_common_factor = extract_most_common_factor(expr)

    assert most_common_factor[0] == 1
    assert sympy.prod(most_common_factor) == expr
    assert most_common_factor[1] == expr


def test_count_operations():
    x, y, z = sympy.symbols('x y z')
    expr = 1/x + y * sympy.sqrt(z)
    ops = count_operations(expr, only_type=None)
    assert ops['adds'] == 1
    assert ops['muls'] == 1
    assert ops['divs'] == 1
    assert ops['sqrts'] == 1

    expr = sympy.sqrt(x + y)
    expr = insert_fast_sqrts(expr).atoms(fast_sqrt)
    ops = count_operations(*expr, only_type=None)
    assert ops['fast_sqrts'] == 1

    expr = sympy.sqrt(x / y)
    expr = insert_fast_divisions(expr).atoms(fast_division)
    ops = count_operations(*expr, only_type=None)
    assert ops['fast_div'] == 1

    expr = pystencils.Assignment(sympy.Symbol('tmp'), 3 / sympy.sqrt(x + y))
    expr = insert_fast_sqrts(expr).atoms(fast_inv_sqrt)
    ops = count_operations(*expr, only_type=None)
    assert ops['fast_inv_sqrts'] == 1

    expr = sympy.Piecewise((1.0, x > 0), (0.0, True)) + y * z
    ops = count_operations(expr, only_type=None)
    assert ops['adds'] == 1

    expr = sympy.Pow(1/x + y * sympy.sqrt(z), 100)
    ops = count_operations(expr, only_type=None)
    assert ops['adds'] == 1
    assert ops['muls'] == 99
    assert ops['divs'] == 1
    assert ops['sqrts'] == 1


def test_common_denominator():
    x = sympy.symbols('x')
    expr = sympy.Rational(1, 2) + x * sympy.Rational(2, 3)
    cm = common_denominator(expr)
    assert cm == 6


def test_get_symmetric_part():
    x, y, z = sympy.symbols('x y z')
    expr = x / 9 - y ** 2 / 6 + z ** 2 / 3 + z / 3
    expected_result = x / 9 - y ** 2 / 6 + z ** 2 / 3
    sym_part = get_symmetric_part(expr, sympy.symbols(f'y z'))

    assert sym_part == expected_result