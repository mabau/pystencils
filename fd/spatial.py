import sympy as sp
from functools import partial
from pystencils import AssignmentCollection, Field
from pystencils.fd import Diff
from .derivative import diff_args


def fd_stencils_standard(indices, dx, fa):
    order = len(indices)
    if order == 1:
        idx = indices[0]
        return (fa.neighbor(idx, 1) - fa.neighbor(idx, -1)) / (2 * dx)
    elif order == 2:
        if indices[0] == indices[1]:
            return (-2 * fa + fa.neighbor(indices[0], -1) + fa.neighbor(indices[0], +1)) / (dx ** 2)
        else:
            offsets = [(1, 1), [-1, 1], [1, -1], [-1, -1]]
            return sum(o1 * o2 * fa.neighbor(indices[0], o1).neighbor(indices[1], o2)
                       for o1, o2 in offsets) / (4 * dx ** 2)
    raise NotImplementedError("Supports only derivatives up to order 2")


def fd_stencils_isotropic(indices, dx, fa):
    dim = fa.field.spatial_dimensions
    if dim == 1:
        return fd_stencils_standard(indices, dx, fa)
    elif dim == 2:
        order = len(indices)

        if order == 1:
            idx = indices[0]
            assert 0 <= idx < 2
            other_idx = 1 if indices[0] == 0 else 0
            weights = {-1: sp.Rational(1, 12) / dx,
                       0: sp.Rational(1, 3) / dx,
                       1: sp.Rational(1, 12) / dx}
            upper_terms = sum(fa.neighbor(idx, +1).neighbor(other_idx, off) * w for off, w in weights.items())
            lower_terms = sum(fa.neighbor(idx, -1).neighbor(other_idx, off) * w for off, w in weights.items())
            return upper_terms - lower_terms
        elif order == 2:
            if indices[0] == indices[1]:
                idx = indices[0]
                other_idx = 1 if idx == 0 else 0
                diagonals = sp.Rational(1, 12) * sum(fa.neighbor(0, i).neighbor(1, j) for i in (-1, 1) for j in (-1, 1))
                div_direction = sp.Rational(5, 6) * sum(fa.neighbor(idx, i) for i in (-1, 1))
                other_direction = - sp.Rational(1, 6) * sum(fa.neighbor(other_idx, i) for i in (-1, 1))
                center = - sp.Rational(5, 3) * fa
                return (diagonals + div_direction + other_direction + center) / (dx ** 2)
            else:
                return fd_stencils_standard(indices, dx, fa)
    raise NotImplementedError("Supports only derivatives up to order 2 for 1D and 2D setups")


def fd_stencils_isotropic_high_density_code(indices, dx, fa):
    dim = fa.field.spatial_dimensions
    if dim == 1:
        return fd_stencils_standard(indices, dx, fa)
    elif dim == 2:
        order = len(indices)

        if order == 1:
            idx = indices[0]
            assert 0 <= idx < 2
            other_idx = 1 if indices[0] == 0 else 0
            weights = {-1: sp.Rational(1, 12) / dx,
                       0: sp.Rational(1, 3) / dx,
                       1: sp.Rational(1, 12) / dx}
            upper_terms = sum(fa.neighbor(idx, +1).neighbor(other_idx, off) * w for off, w in weights.items())
            lower_terms = sum(fa.neighbor(idx, -1).neighbor(other_idx, off) * w for off, w in weights.items())
            return upper_terms - lower_terms
        elif order == 2:
            if indices[0] == indices[1]:
                idx = indices[0]
                diagonals = sp.Rational(1, 8) * sum(fa.neighbor(0, i).neighbor(1, j) for i in (-1, 1) for j in (-1, 1))
                div_direction = sp.Rational(1, 2) * sum(fa.neighbor(idx, i) for i in (-1, 1))
                center = - sp.Rational(3, 2) * fa
                return (diagonals + div_direction + center) / (dx ** 2)
            else:
                return fd_stencils_standard(indices, dx, fa)
    raise NotImplementedError("Supports only derivatives up to order 2 for 1D and 2D setups")


def discretize_spatial(expr, dx, stencil=fd_stencils_standard):
    if isinstance(stencil, str):
        if stencil == 'standard':
            stencil = fd_stencils_standard
        elif stencil == 'isotropic':
            stencil = fd_stencils_isotropic
        else:
            raise ValueError("Unknown stencil. Supported 'standard' and 'isotropic'")

    if isinstance(expr, list):
        return [discretize_spatial(e, dx, stencil) for e in expr]
    elif isinstance(expr, sp.Matrix):
        return expr.applyfunc(partial(discretize_spatial, dx=dx, stencil=stencil))
    elif isinstance(expr, AssignmentCollection):
        return expr.copy(main_assignments=[e for e in expr.main_assignments],
                         subexpressions=[e for e in expr.subexpressions])
    elif isinstance(expr, Diff):
        arg, *indices = diff_args(expr)
        if not isinstance(arg, Field.Access):
            raise ValueError("Only derivatives with field or field accesses as arguments can be discretized")
        return stencil(indices, dx, arg)
    else:
        new_args = [discretize_spatial(a, dx, stencil) for a in expr.args]
        return expr.func(*new_args) if new_args else expr
