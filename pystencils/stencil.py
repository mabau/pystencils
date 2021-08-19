"""This submodule offers functions to work with stencils in expression an offset-list form."""
from collections import defaultdict
from typing import Sequence

import numpy as np
import sympy as sp


def inverse_direction(direction):
    """Returns inverse i.e. negative of given direction tuple

    Example:
        >>> inverse_direction((1, -1, 0))
        (-1, 1, 0)
    """
    return tuple([-i for i in direction])


def inverse_direction_string(direction):
    """Returns inverse of given direction string"""
    return offset_to_direction_string(inverse_direction(direction_string_to_offset(direction)))


def is_valid(stencil, max_neighborhood=None):
    """
    Tests if a nested sequence is a valid stencil i.e. all the inner sequences have the same length.
    If max_neighborhood is specified, it is also verified that the stencil does not contain any direction components
    with absolute value greater than the maximal neighborhood.

    Examples:
        >>> is_valid([(1, 0), (1, 0, 0)])  # stencil entries have different length
        False
        >>> is_valid([(2, 0), (1, 0)])
        True
        >>> is_valid([(2, 0), (1, 0)], max_neighborhood=1)
        False
        >>> is_valid([(2, 0), (1, 0)], max_neighborhood=2)
        True
    """
    expected_dim = len(stencil[0])
    for d in stencil:
        if len(d) != expected_dim:
            return False
        if max_neighborhood is not None:
            for d_i in d:
                if abs(d_i) > max_neighborhood:
                    return False
    return True


def is_symmetric(stencil):
    """Tests for every direction d, that -d is also in the stencil

    Examples:
        >>> is_symmetric([(1, 0), (0, 1)])
        False
        >>> is_symmetric([(1, 0), (-1, 0)])
        True
    """
    for d in stencil:
        if inverse_direction(d) not in stencil:
            return False
    return True


def have_same_entries(s1, s2):
    """Checks if two stencils are the same

    Examples:
        >>> stencil1 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        >>> stencil2 = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        >>> stencil3 = [(-1, 0), (0, -1), (1, 0)]
        >>> have_same_entries(stencil1, stencil2)
        True
        >>> have_same_entries(stencil1, stencil3)
        False
    """
    if len(s1) != len(s2):
        return False
    return len(set(s1) - set(s2)) == 0


# -------------------------------------Expression - Coefficient Form Conversion ----------------------------------------


def coefficient_dict(expr):
    """Extracts coefficients in front of field accesses in a expression.

    Expression may only access a single field at a single index.

    Returns:
        center, coefficient dict, nonlinear part
        where center is the single field that is accessed in expression accessed at center
        and coefficient dict maps offsets to coefficients. The nonlinear part is everything that is not in the form of
        coefficient times field access.

    Examples:
        >>> import pystencils as ps
        >>> f = ps.fields("f(3) : double[2D]")
        >>> field, coeffs, nonlinear_part = coefficient_dict(2 * f[0, 1](1) + 3 * f[-1, 0](1) + 123)
        >>> assert nonlinear_part == 123 and field == f(1)
        >>> sorted(coeffs.items())
        [((-1, 0), 3), ((0, 1), 2)]
    """
    from pystencils.field import Field
    expr = expr.expand()
    field_accesses = expr.atoms(Field.Access)
    fields = set(fa.field for fa in field_accesses)
    accessed_indices = set(fa.index for fa in field_accesses)

    if len(fields) != 1:
        raise ValueError("Could not extract stencil coefficients. "
                         "Expression has to be a linear function of exactly one field.")
    if len(accessed_indices) != 1:
        raise ValueError("Could not extract stencil coefficients. Field is accessed at multiple indices")

    field = fields.pop()
    idx = accessed_indices.pop()

    coeffs = defaultdict(lambda: 0)
    coeffs.update({fa.offsets: expr.coeff(fa) for fa in field_accesses})

    linear_part = sum(c * field[off](*idx) for off, c in coeffs.items())
    nonlinear_part = expr - linear_part
    return field(*idx), coeffs, nonlinear_part


def coefficients(expr):
    """Returns two lists - one with accessed offsets and one with their coefficients.

    Same restrictions as `coefficient_dict` apply. Expression must not have any nonlinear part

    >>> import pystencils as ps
    >>> f = ps.fields("f(3) : double[2D]")
    >>> coff = coefficients(2 * f[0, 1](1) + 3 * f[-1, 0](1))
    """
    field_center, coeffs, nonlinear_part = coefficient_dict(expr)
    assert nonlinear_part == 0
    stencil = list(coeffs.keys())
    entries = [coeffs[c] for c in stencil]
    return stencil, entries


def coefficient_list(expr, matrix_form=False):
    """Returns stencil coefficients in the form of nested lists

    Same restrictions as `coefficient_dict` apply. Expression must not have any nonlinear part

    Examples:
        >>> import pystencils as ps
        >>> f = ps.fields("f: double[2D]")
        >>> coefficient_list(2 * f[0, 1] + 3 * f[-1, 0])
        [[0, 0, 0], [3, 0, 0], [0, 2, 0]]
        >>> coefficient_list(2 * f[0, 1] + 3 * f[-1, 0], matrix_form=True)
        Matrix([
        [0, 2, 0],
        [3, 0, 0],
        [0, 0, 0]])
    """
    field_center, coeffs, nonlinear_part = coefficient_dict(expr)
    assert nonlinear_part == 0
    field = field_center.field

    dim = field.spatial_dimensions
    max_offsets = defaultdict(lambda: 0)
    for offset in coeffs.keys():
        for d, off in enumerate(offset):
            max_offsets[d] = max(max_offsets[d], abs(off))

    if dim == 1:
        result = [coeffs[(i,)] for i in range(-max_offsets[0], max_offsets[0] + 1)]
        return sp.Matrix(result) if matrix_form else result
    else:
        y_range = list(range(-max_offsets[1], max_offsets[1] + 1))
        if matrix_form:
            y_range.reverse()
        if dim == 2:
            result = [[coeffs[(i, j)]
                       for i in range(-max_offsets[0], max_offsets[0] + 1)]
                      for j in y_range]
            return sp.Matrix(result) if matrix_form else result
        elif dim == 3:
            result = [[[coeffs[(i, j, k)]
                        for i in range(-max_offsets[0], max_offsets[0] + 1)]
                       for j in y_range]
                      for k in range(-max_offsets[2], max_offsets[2] + 1)]
            return [sp.Matrix(l) for l in result] if matrix_form else result
        else:
            raise ValueError("Can only handle fields with 1,2 or 3 spatial dimensions")


# ------------------------------------- Point-on-compass notation ------------------------------------------------------


def offset_component_to_direction_string(coordinate_id: int, value: int) -> str:
    """Translates numerical offset to string notation.

    x offsets are labeled with east 'E' and 'W',
    y offsets with north 'N' and 'S' and
    z offsets with top 'T' and bottom 'B'
    If the absolute value of the offset is bigger than 1, this number is prefixed.

    Args:
        coordinate_id: integer 0, 1 or 2 standing for x,y and z
        value: integer offset

    Examples:
        >>> offset_component_to_direction_string(0, 1)
        'E'
        >>> offset_component_to_direction_string(1, 2)
        '2N'
    """
    assert 0 <= coordinate_id < 3, "Works only for at most 3D arrays"
    name_components = (('W', 'E'),  # west, east
                       ('S', 'N'),  # south, north
                       ('B', 'T'))  # bottom, top
    if value == 0:
        result = ""
    elif value < 0:
        result = name_components[coordinate_id][0]
    else:
        result = name_components[coordinate_id][1]
    if abs(value) > 1:
        result = "%d%s" % (abs(value), result)
    return result


def offset_to_direction_string(offsets: Sequence[int]) -> str:
    """
    Translates numerical offset to string notation.
    For details see :func:`offset_component_to_direction_string`
    Args:
        offsets: 3-tuple with x,y,z offset

    Examples:
        >>> offset_to_direction_string([1, -1, 0])
        'SE'
        >>> offset_to_direction_string(([-3, 0, -2]))
        '2B3W'
    """
    if len(offsets) > 3:
        return str(offsets)
    names = ["", "", ""]
    for i in range(len(offsets)):
        names[i] = offset_component_to_direction_string(i, offsets[i])
    name = "".join(reversed(names))
    if name == "":
        name = "C"
    return name


def direction_string_to_offset(direction: str, dim: int = 3):
    """
    Reverse mapping of :func:`offset_to_direction_string`

    Args:
        direction: string representation of offset
        dim: dimension of offset, i.e the length of the returned list

    Examples:
        >>> direction_string_to_offset('NW', dim=3)
        array([-1,  1,  0])
        >>> direction_string_to_offset('NW', dim=2)
        array([-1,  1])
        >>> direction_string_to_offset(offset_to_direction_string((3,-2,1)))
        array([ 3, -2,  1])
    """
    offset_dict = {
        'C': np.array([0, 0, 0]),

        'W': np.array([-1, 0, 0]),
        'E': np.array([1, 0, 0]),

        'S': np.array([0, -1, 0]),
        'N': np.array([0, 1, 0]),

        'B': np.array([0, 0, -1]),
        'T': np.array([0, 0, 1]),
    }
    offset = np.array([0, 0, 0])

    while len(direction) > 0:
        factor = 1
        first_non_digit = 0
        while direction[first_non_digit].isdigit():
            first_non_digit += 1
        if first_non_digit > 0:
            factor = int(direction[:first_non_digit])
            direction = direction[first_non_digit:]
        cur_offset = offset_dict[direction[0]]
        offset += factor * cur_offset
        direction = direction[1:]
    return offset[:dim]


# -------------------------------------- Visualization -----------------------------------------------------------------


def plot(stencil, **kwargs):
    dim = len(stencil[0])
    if dim == 2:
        plot_2d(stencil, **kwargs)
    else:
        slicing = False
        if 'slice' in kwargs:
            slicing = kwargs['slice']
            del kwargs['slice']

        if slicing:
            plot_3d_slicing(stencil, **kwargs)
        else:
            plot_3d(stencil, **kwargs)


def plot_2d(stencil, axes=None, figure=None, data=None, textsize='12', **kwargs):
    """
    Creates a matplotlib 2D plot of the stencil

    Args:
        stencil: sequence of directions
        axes: optional matplotlib axes
        figure: optional matplotlib figure
        data: data to annotate the directions with, if none given, the indices are used
        textsize: size of annotation text
    """
    from matplotlib.patches import BoxStyle
    import matplotlib.pyplot as plt

    if axes is None:
        if figure is None:
            figure = plt.gcf()
        axes = figure.gca()

    text_box_style = BoxStyle("Round", pad=0.3)
    head_length = 0.1
    max_offsets = [max(abs(int(d[c])) for d in stencil) for c in (0, 1)]

    if data is None:
        data = list(range(len(stencil)))

    for direction, annotation in zip(stencil, data):
        assert len(direction) == 2, "Works only for 2D stencils"
        direction = tuple(int(i) for i in direction)
        if not(direction[0] == 0 and direction[1] == 0):
            axes.arrow(0, 0, direction[0], direction[1], head_width=0.08, head_length=head_length, color='k')

        if isinstance(annotation, sp.Basic):
            annotation = "$" + sp.latex(annotation) + "$"
        else:
            annotation = str(annotation)

        def position_correction(d, magnitude=0.18):
            if d < 0:
                return -magnitude
            elif d > 0:
                return +magnitude
            else:
                return 0
        text_position = [direction[c] + position_correction(direction[c]) for c in (0, 1)]
        axes.text(x=text_position[0], y=text_position[1], s=annotation, verticalalignment='center',
                  zorder=30, horizontalalignment='center', size=textsize,
                  bbox=dict(boxstyle=text_box_style, facecolor='#00b6eb', alpha=0.85, linewidth=0))

    axes.set_axis_off()
    axes.set_aspect('equal')
    max_offsets = [m if m > 0 else 0.1 for m in max_offsets]
    border = 0.1
    axes.set_xlim([-border - max_offsets[0], border + max_offsets[0]])
    axes.set_ylim([-border - max_offsets[1], border + max_offsets[1]])


def plot_3d_slicing(stencil, slice_axis=2, figure=None, data=None, **kwargs):
    """Visualizes a 3D, first-neighborhood stencil by plotting 3 slices along a given axis.

    Args:
        stencil: stencil as sequence of directions
        slice_axis: 0, 1, or 2 indicating the axis to slice through
        figure: optional matplotlib figure
        data: optional data to print as text besides the arrows
    """
    import matplotlib.pyplot as plt

    for d in stencil:
        for element in d:
            assert element == -1 or element == 0 or element == 1, "This function can only first neighborhood stencils"

    if figure is None:
        figure = plt.gcf()

    axes = [figure.add_subplot(1, 3, i + 1) for i in range(3)]
    splitted_directions = [[], [], []]
    splitted_data = [[], [], []]
    axes_names = ['x', 'y', 'z']

    for i, d in enumerate(stencil):
        split_idx = d[slice_axis] + 1
        reduced_dir = tuple([element for j, element in enumerate(d) if j != slice_axis])
        splitted_directions[split_idx].append(reduced_dir)
        splitted_data[split_idx].append(i if data is None else data[i])

    for i in range(3):
        plot_2d(splitted_directions[i], axes=axes[i], data=splitted_data[i], **kwargs)
    for i in [-1, 0, 1]:
        axes[i + 1].set_title("Cut at %s=%d" % (axes_names[slice_axis], i), y=1.08)


def plot_3d(stencil, figure=None, axes=None, data=None, textsize='8'):
    """
    Draws 3D stencil into a 3D coordinate system, parameters are similar to :func:`visualize_stencil_2d`
    If data is None, no labels are drawn. To draw the labels as in the 2D case, use ``data=list(range(len(stencil)))``
    """
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    import matplotlib.pyplot as plt
    from matplotlib.patches import BoxStyle
    from itertools import product, combinations
    import numpy as np

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    if axes is None:
        if figure is None:
            figure = plt.figure()
        axes = figure.add_subplot(projection='3d')
        try:
            axes.set_aspect("equal")
        except NotImplementedError:
            pass

    if data is None:
        data = [None] * len(stencil)

    text_offset = 1.25
    text_box_style = BoxStyle("Round", pad=0.3)

    # Draw cell (cube)
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            axes.plot(*zip(s, e), color="k", alpha=0.5)

    for d, annotation in zip(stencil, data):
        assert len(d) == 3, "Works only for 3D stencils"
        d = tuple(int(i) for i in d)
        if not (d[0] == 0 and d[1] == 0 and d[2] == 0):
            if d[0] == 0:
                color = '#348abd'
            elif d[1] == 0:
                color = '#fac364'
            elif sum([abs(d) for d in d]) == 2:
                color = '#95bd50'
            else:
                color = '#808080'

            a = Arrow3D([0, d[0]], [0, d[1]], [0, d[2]], mutation_scale=20, lw=2, arrowstyle="-|>", color=color)
            axes.add_artist(a)

        if annotation:
            if isinstance(annotation, sp.Basic):
                annotation = "$" + sp.latex(annotation) + "$"
            else:
                annotation = str(annotation)

            axes.text(x=d[0] * text_offset, y=d[1] * text_offset, z=d[2] * text_offset,
                      s=annotation, verticalalignment='center', zorder=30,
                      size=textsize, bbox=dict(boxstyle=text_box_style, facecolor='#777777', alpha=0.6, linewidth=0))

    axes.set_xlim([-text_offset * 1.1, text_offset * 1.1])
    axes.set_ylim([-text_offset * 1.1, text_offset * 1.1])
    axes.set_zlim([-text_offset * 1.1, text_offset * 1.1])
    axes.set_axis_off()


def plot_expression(expr, **kwargs):
    """Displays coefficients of a linear update expression of a single field as matplotlib arrow drawing."""
    stencil, coeffs = coefficients(expr)
    dim = len(stencil[0])
    assert 0 < dim <= 3
    if dim == 1:
        return coefficient_list(expr, matrix_form=True)
    elif dim == 2:
        return plot_2d(stencil, data=coeffs, **kwargs)
    elif dim == 3:
        return plot_3d_slicing(stencil, data=coeffs, **kwargs)
