from functools import lru_cache
from typing import Tuple

import sympy as sp

from pystencils.astnodes import LoopOverCoordinate
from pystencils.fd import Diff
from pystencils.field import Field
from pystencils.transformations import generic_visit

from .derivation import FiniteDifferenceStencilDerivation
from .derivative import diff_args


def fd_stencils_standard(indices, dx, fa):
    order = len(indices)
    assert all(i >= 0 for i in indices), "Can only discretize objects with (integer) subscripts"
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


def fd_stencils_forth_order_isotropic(indices, dx, fa):
    order = len(indices)
    if order != 1:
        raise NotImplementedError("Forth order finite difference discretization is "
                                  "currently only supported for first derivatives")
    dim = indices[0]
    if dim not in (0, 1):
        raise NotImplementedError("Forth order finite difference discretization is only implemented for 2D")

    stencils = forth_order_2d_derivation()
    return stencils[dim].apply(fa) / dx


def discretize_spatial(expr, dx, stencil=fd_stencils_standard):
    if isinstance(stencil, str):
        if stencil == 'standard':
            stencil = fd_stencils_standard
        elif stencil == 'isotropic':
            stencil = fd_stencils_isotropic
        else:
            raise ValueError("Unknown stencil. Supported 'standard' and 'isotropic'")

    def visitor(e):
        if isinstance(e, Diff):
            arg, *indices = diff_args(e)
            if not isinstance(arg, Field.Access):
                raise ValueError("Only derivatives with field or field accesses as arguments can be discretized")
            return stencil(indices, dx, arg)
        else:
            new_args = [discretize_spatial(a, dx, stencil) for a in e.args]
            return e.func(*new_args) if new_args else e

    return generic_visit(expr, visitor)


def discretize_spatial_staggered(expr, dx, stencil=fd_stencils_standard):
    def staggered_visitor(e, coordinate, sign):
        if isinstance(e, Diff):
            arg, *indices = diff_args(e)
            if len(indices) != 1:
                raise ValueError("Function supports only up to second derivatives")
            if not isinstance(arg, Field.Access):
                raise ValueError("Argument of inner derivative has to be field access")
            target = indices[0]
            if target == coordinate:
                assert sign in (-1, 1)
                return (arg.neighbor(coordinate, sign) - arg) / dx * sign
            else:
                return (stencil(indices, dx, arg.neighbor(coordinate, sign))
                        + stencil(indices, dx, arg)) / 2
        elif isinstance(e, Field.Access):
            return (e.neighbor(coordinate, sign) + e) / 2
        elif isinstance(e, sp.Symbol):
            loop_idx = LoopOverCoordinate.is_loop_counter_symbol(e)
            return e + sign / 2 if loop_idx == coordinate else e
        else:
            new_args = [staggered_visitor(a, coordinate, sign) for a in e.args]
            return e.func(*new_args) if new_args else e

    def visitor(e):
        if isinstance(e, Diff):
            arg, *indices = diff_args(e)
            if isinstance(arg, Field.Access):
                return stencil(indices, dx, arg)
            else:
                if not len(indices) == 1:
                    raise ValueError("This term is not support by the staggered discretization strategy")
                target = indices[0]
                return (staggered_visitor(arg, target, 1) - staggered_visitor(arg, target, -1)) / dx
        else:
            new_args = [visitor(a) for a in e.args]
            return e.func(*new_args) if new_args else e

    return generic_visit(expr, visitor)


# -------------------------------------- special stencils --------------------------------------------------------------
@lru_cache(maxsize=1)
def forth_order_2d_derivation() -> Tuple[FiniteDifferenceStencilDerivation.Result, ...]:
    # Symmetry, isotropy and 4th order conditions are not enough to fully specify the stencil
    # one weight has to be specifically set to a somewhat arbitrary value
    second_neighbor_weight = sp.Rational(1, 10)
    second_neighbor_stencil = [(i, j)
                               for i in (-2, -1, 0, 1, 2)
                               for j in (-2, -1, 0, 1, 2)
                               ]
    x_diff = FiniteDifferenceStencilDerivation((0,), second_neighbor_stencil)
    x_diff.set_weight((2, 0), second_neighbor_weight)
    x_diff.assume_symmetric(0, anti_symmetric=True)
    x_diff.assume_symmetric(1)
    x_diff_stencil = x_diff.get_stencil(isotropic=True)

    y_diff = FiniteDifferenceStencilDerivation((1,), second_neighbor_stencil)
    y_diff.set_weight((0, 2), second_neighbor_weight)
    y_diff.assume_symmetric(1, anti_symmetric=True)
    y_diff.assume_symmetric(0)
    y_diff_stencil = y_diff.get_stencil(isotropic=True)

    return x_diff_stencil, y_diff_stencil
