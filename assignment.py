# -*- coding: utf-8 -*-
import sympy as sp
from sympy.printing.latex import LatexPrinter

try:
    from sympy.codegen.ast import Assignment
except ImportError:
    Assignment = None
import numpy as np

__all__ = ['Assignment', 'assignment_from_stencil']


def print_assignment_latex(printer, expr):
    """sympy cannot print Assignments as Latex. Thus, this function is added to the sympy Latex printer"""
    printed_lhs = printer.doprint(expr.lhs)
    printed_rhs = printer.doprint(expr.rhs)
    return "{printed_lhs} \leftarrow {printed_rhs}".format(printed_lhs=printed_lhs, printed_rhs=printed_rhs)


def assignment_str(assignment):
    return "{lhs} ← {rhs}".format(lhs=assignment.lhs, rhs=assignment.rhs)


if Assignment:

    Assignment.__str__ = assignment_str
    LatexPrinter._print_Assignment = print_assignment_latex

else:
    # back port for older sympy versions that don't have Assignment  yet

    class Assignment(sp.Rel):  # pragma: no cover

        rel_op = ':='
        __slots__ = []

        def __new__(cls, lhs, rhs=0, **assumptions):
            from sympy.matrices.expressions.matexpr import (
                MatrixElement, MatrixSymbol)
            from sympy.tensor.indexed import Indexed
            lhs = sp.sympify(lhs)
            rhs = sp.sympify(rhs)
            # Tuple of things that can be on the lhs of an assignment
            assignable = (sp.Symbol, MatrixSymbol, MatrixElement, Indexed)
            if not isinstance(lhs, assignable):
                raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
            return sp.Rel.__new__(cls, lhs, rhs, **assumptions)

        __str__ = assignment_str
        _print_Assignment = print_assignment_latex


def assignment_from_stencil(stencil_array, input_field, output_field, normalization_factor=None):
    stencil_array = np.array(stencil_array)
    rhs = 0
    offset = tuple(s // 2 for s in stencil_array.shape)

    for index, factor in np.ndenumerate(stencil_array):
        rhs += factor * input_field[tuple(i - o for i, o in zip(index, offset))]

    if normalization_factor:
        rhs *= normalization_factor

    return Assignment(output_field.center(), rhs)
