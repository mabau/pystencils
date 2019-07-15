# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp
from sympy.printing.latex import LatexPrinter

try:
    from sympy.codegen.ast import Assignment
except ImportError:
    Assignment = None

__all__ = ['Assignment', 'assignment_from_stencil']


def print_assignment_latex(printer, expr):
    """sympy cannot print Assignments as Latex. Thus, this function is added to the sympy Latex printer"""
    printed_lhs = printer.doprint(expr.lhs)
    printed_rhs = printer.doprint(expr.rhs)
    return r"{printed_lhs} \leftarrow {printed_rhs}".format(printed_lhs=printed_lhs, printed_rhs=printed_rhs)


def assignment_str(assignment):
    return r"{lhs} ← {rhs}".format(lhs=assignment.lhs, rhs=assignment.rhs)


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
            lhs = sp.sympify(lhs)
            rhs = sp.sympify(rhs)
            # Tuple of things that can be on the lhs of an assignment
            assignable = (sp.Symbol, MatrixSymbol, MatrixElement, sp.Indexed)
            if not isinstance(lhs, assignable):
                raise TypeError("Cannot assign to lhs of type %s." % type(lhs))
            return sp.Rel.__new__(cls, lhs, rhs, **assumptions)

        __str__ = assignment_str
        _print_Assignment = print_assignment_latex


def assignment_from_stencil(stencil_array, input_field, output_field,
                            normalization_factor=None, order='visual') -> Assignment:
    """Creates an assignment

    Args:
        stencil_array: nested list of numpy array defining the stencil weights
        input_field: field or field access, defining where the stencil should be applied to
        output_field: field or field access where the result is written to
        normalization_factor: optional normalization factor for the stencil
        order: defines how the stencil_array is interpreted. Possible values are 'visual' and 'numpy'.
               For details see examples

    Returns:
        Assignment that can be used to create a kernel

    Examples:
        >>> import pystencils as ps
        >>> f, g = ps.fields("f, g: [2D]")
        >>> stencil = [[0, 2, 0],
        ...            [3, 4, 5],
        ...            [0, 6, 0]]

        By default 'visual ordering is used - i.e. the stencil is applied as the nested lists are written down
        >>> assignment_from_stencil(stencil, f, g, order='visual')
        Assignment(g_C, 3*f_W + 6*f_S + 4*f_C + 2*f_N + 5*f_E)

        'numpy' ordering uses the first coordinate of the stencil array for x offset, second for y offset etc.
        >>> assignment_from_stencil(stencil, f, g, order='numpy')
        Assignment(g_C, 2*f_W + 3*f_S + 4*f_C + 5*f_N + 6*f_E)

        You can also pass field accesses to apply the stencil at an already shifted position:
        >>> assignment_from_stencil(stencil, f[1, 0], g[2, 0])
        Assignment(g_2E, 3*f_C + 6*f_SE + 4*f_E + 2*f_NE + 5*f_2E)
    """
    from pystencils.field import Field

    stencil_array = np.array(stencil_array)
    if order == 'visual':
        stencil_array = np.swapaxes(stencil_array, 0, 1)
        stencil_array = np.flip(stencil_array, axis=1)
    elif order == 'numpy':
        pass
    else:
        raise ValueError("'order' has to be either 'visual' or 'numpy'")

    if isinstance(input_field, Field):
        input_field = input_field.center
    if isinstance(output_field, Field):
        output_field = output_field.center

    rhs = 0
    offset = tuple(s // 2 for s in stencil_array.shape)

    for index, factor in np.ndenumerate(stencil_array):
        shift = tuple(i - o for i, o in zip(index, offset))
        rhs += factor * input_field.get_shifted(*shift)

    if normalization_factor:
        rhs *= normalization_factor

    return Assignment(output_field, rhs)
