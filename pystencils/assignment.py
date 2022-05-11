import numpy as np
import sympy as sp
from sympy.codegen.ast import Assignment
from sympy.printing.latex import LatexPrinter

__all__ = ['Assignment', 'assignment_from_stencil']


def print_assignment_latex(printer, expr):
    """sympy cannot print Assignments as Latex. Thus, this function is added to the sympy Latex printer"""
    printed_lhs = printer.doprint(expr.lhs)
    printed_rhs = printer.doprint(expr.rhs)
    return fr"{printed_lhs} \leftarrow {printed_rhs}"


def assignment_str(assignment):
    return fr"{assignment.lhs} ← {assignment.rhs}"


_old_new = sp.codegen.ast.Assignment.__new__


# TODO Typing Part2 add default type, defult_float_type, default_int_type and use sane defaults
def _Assignment__new__(cls, lhs, rhs, *args, **kwargs):
    if isinstance(lhs, (list, tuple, sp.Matrix)) and isinstance(rhs, (list, tuple, sp.Matrix)):
        assert len(lhs) == len(rhs), f'{lhs} and {rhs} must have same length when performing vector assignment!'
        return tuple(_old_new(cls, a, b, *args, **kwargs) for a, b in zip(lhs, rhs))
    return _old_new(cls, lhs, rhs, *args, **kwargs)


Assignment.__str__ = assignment_str
Assignment.__new__ = _Assignment__new__
LatexPrinter._print_Assignment = print_assignment_latex

sp.MutableDenseMatrix.__hash__ = lambda self: hash(tuple(self))


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
        >>> expected_output = Assignment(g[0, 0], 3*f[-1, 0] + 6*f[0, -1] + 4*f[0, 0] + 2*f[0, 1] + 5*f[1, 0])
        >>> assignment_from_stencil(stencil, f, g, order='visual') == expected_output
        True

        'numpy' ordering uses the first coordinate of the stencil array for x offset, second for y offset etc.
        >>> expected_output = Assignment(g[0, 0], 2*f[-1, 0] + 3*f[0, -1] + 4*f[0, 0] + 5*f[0, 1] + 6*f[1, 0])
        >>> assignment_from_stencil(stencil, f, g, order='numpy') == expected_output
        True

        You can also pass field accesses to apply the stencil at an already shifted position:
        >>> expected_output = Assignment(g[2, 0], 3*f[0, 0] + 6*f[1, -1] + 4*f[1, 0] + 2*f[1, 1] + 5*f[2, 0])
        >>> assignment_from_stencil(stencil, f[1, 0], g[2, 0]) == expected_output
        True
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
