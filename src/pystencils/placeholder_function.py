from typing import List

import sympy as sp

from pystencils.assignment import Assignment
from pystencils.astnodes import Node
from pystencils.sympyextensions import is_constant
from pystencils.transformations import generic_visit


class PlaceholderFunction:
    pass


def to_placeholder_function(expr, name):
    """Replaces an expression by a sympy function.

    - replacing an expression with just a symbol would lead to problem when calculating derivatives
    - placeholder functions get rid of this problem

    Examples:
        >>> x, t = sp.symbols("x, t")
        >>> temperature = x**2 + t**4 # some 'complicated' dependency
        >>> temperature_placeholder = to_placeholder_function(temperature, 'T')
        >>> diffusivity = temperature_placeholder + 42 * t
        >>> sp.diff(diffusivity, t)  # returns a symbol instead of the computed derivative
        _dT_dt + 42
        >>> result, subexpr = remove_placeholder_functions(diffusivity)
        >>> result
        T + 42*t
        >>> subexpr
        [Assignment(T, t**4 + x**2), Assignment(_dT_dt, 4*t**3), Assignment(_dT_dx, 2*x)]

    """
    symbols = list(expr.atoms(sp.Symbol))
    symbols.sort(key=lambda e: e.name)
    derivative_symbols = [sp.Symbol(f"_d{name}_d{s.name}") for s in symbols]
    derivatives = [sp.diff(expr, s) for s in symbols]

    assignments = [Assignment(sp.Symbol(name), expr)]
    assignments += [Assignment(symbol, derivative)
                    for symbol, derivative in zip(derivative_symbols, derivatives)
                    if not is_constant(derivative)]

    def fdiff(_, index):
        result = derivatives[index - 1]
        return result if is_constant(result) else derivative_symbols[index - 1]

    func = type(name, (sp.Function, PlaceholderFunction),
                {'fdiff': fdiff,
                 'value': sp.Symbol(name),
                 'subexpressions': assignments,
                 'nargs': len(symbols)})
    return func(*symbols)


def remove_placeholder_functions(expr):
    subexpressions = []

    def visit(e):
        if isinstance(e, Node):
            return e
        elif isinstance(e, PlaceholderFunction):
            for se in e.subexpressions:
                if se.lhs not in {a.lhs for a in subexpressions}:
                    subexpressions.append(se)
            return e.value
        else:
            new_args = [visit(a) for a in e.args]
            return e.func(*new_args) if new_args else e

    return generic_visit(expr, visit), subexpressions


def prepend_placeholder_functions(assignments: List[Assignment]):
    result, subexpressions = remove_placeholder_functions(assignments)
    return subexpressions + result
