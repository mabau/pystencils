from typing import Optional, Union

import numpy as np
import sympy as sp

from pystencils.fd import Diff
from pystencils.fd.derivative import diff_args
from pystencils.fd.spatial import fd_stencils_standard
from pystencils.field import Field
from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils.sympyextensions import fast_subs

FieldOrFieldAccess = Union[Field, Field.Access]


# --------------------------------------- Advection Diffusion ----------------------------------------------------------


def diffusion(scalar, diffusion_coeff, idx=None):
    """Diffusion term ∇·( diffusion_coeff · ∇(scalar))

    Examples:
        >>> f = Field.create_generic('f', spatial_dimensions=2)
        >>> d = sp.Symbol("d")
        >>> dx = sp.Symbol("dx")
        >>> diffusion_term = diffusion(scalar=f, diffusion_coeff=d)
        >>> discretization = Discretization2ndOrder()
        >>> expected_output = ((f[-1, 0] + f[0, -1] - 4 * f[0, 0] + f[0, 1] + f[1, 0]) * d) / dx**2
        >>> sp.simplify(discretization(diffusion_term) - expected_output)
        0
    """
    if isinstance(scalar, Field):
        first_arg = scalar.center
    elif isinstance(scalar, Field.Access):
        first_arg = scalar
    else:
        raise ValueError("Diffused scalar has to be a pystencils Field or Field.Access")

    args = [first_arg, diffusion_coeff if not isinstance(diffusion_coeff, Field) else diffusion_coeff.center]
    if idx is not None:
        args.append(idx)
    return Diffusion(*args)


def advection(advected_scalar: FieldOrFieldAccess, velocity_field: FieldOrFieldAccess, idx: Optional[int] = None):
    """Advection term  ∇·(velocity_field · advected_scalar)

    Term that describes the advection of a scalar quantity in a velocity field.
    """
    if isinstance(advected_scalar, Field):
        first_arg = advected_scalar.center
    elif isinstance(advected_scalar, Field.Access):
        first_arg = advected_scalar
    else:
        raise ValueError("Advected scalar has to be a pystencils Field or Field.Access")

    args = [first_arg, velocity_field if not isinstance(velocity_field, Field) else velocity_field.center]
    if idx is not None:
        args.append(idx)
    return Advection(*args)


def transient(scalar, idx=None):
    """Transient term ∂_t(scalar)"""
    if isinstance(scalar, Field):
        args = [scalar.center]
    elif isinstance(scalar, Field.Access):
        args = [scalar]
    else:
        raise ValueError("Scalar has to be a pystencils Field or Field.Access")
    if idx is not None:
        args.append(idx)
    return Transient(*args)


class Discretization2ndOrder:
    def __init__(self, dx=sp.Symbol("dx"), dt=sp.Symbol("dt"), discretization_stencil_func=fd_stencils_standard):
        self.dx = dx
        self.dt = dt
        self.spatial_stencil = discretization_stencil_func

    def _discretize_diffusion(self, e):
        result = 0
        for c in range(e.dim):
            first_diffs = [offset
                           * (e.diffusion_scalar_at_offset(c, offset) * e.diffusion_coefficient_at_offset(c, offset)
                              - e.diffusion_scalar_at_offset(0, 0) * e.diffusion_coefficient_at_offset(0, 0))
                           for offset in [-1, 1]]
            result += first_diffs[1] - first_diffs[0]
        return result / (self.dx ** 2)

    def _discretize_advection(self, expr):
        result = 0
        for c in range(expr.dim):
            interpolated = [(expr.advected_scalar_at_offset(c, offset) * expr.velocity_field_at_offset(c, offset, c)
                             + expr.advected_scalar_at_offset(c, 0) * expr.velocity_field_at_offset(c, 0, c)) / 2
                            for offset in [-1, 1]]
            result += interpolated[1] - interpolated[0]
        return result / self.dx

    def _discretize_spatial(self, e):
        if isinstance(e, Diffusion):
            return self._discretize_diffusion(e)
        elif isinstance(e, Advection):
            return self._discretize_advection(e)
        elif isinstance(e, Diff):
            arg, *indices = diff_args(e)

            if not isinstance(arg, Field.Access):
                raise ValueError("Only derivatives with field or field accesses as arguments can be discretized")
            return self.spatial_stencil(indices, self.dx, arg)
        else:
            new_args = [self._discretize_spatial(a) for a in e.args]
            return e.func(*new_args) if new_args else e

    def __call__(self, expr):
        if isinstance(expr, list):
            return [self(e) for e in expr]
        elif isinstance(expr, sp.Matrix) or isinstance(expr, sp.ImmutableDenseMatrix):
            return expr.applyfunc(self.__call__)
        elif isinstance(expr, AssignmentCollection):
            return expr.copy(main_assignments=[e for e in expr.main_assignments],
                             subexpressions=[e for e in expr.subexpressions])

        transient_terms = expr.atoms(Transient)
        if len(transient_terms) == 0:
            return self._discretize_spatial(expr)
        elif len(transient_terms) == 1:
            transient_term = transient_terms.pop()
            solve_result = sp.solve(expr, transient_term)
            if len(solve_result) != 1:
                raise ValueError("Could not solve for transient term" + str(solve_result))
            rhs = solve_result.pop()
            # explicit euler
            return transient_term.scalar + self.dt * self._discretize_spatial(rhs)
        else:
            print(transient_terms)
            raise NotImplementedError("Cannot discretize expression with more than one transient term")


# -------------------------------------- Helper Classes ----------------------------------------------------------------

class Advection(sp.Function):

    @property
    def scalar(self):
        return self.args[0].field

    @property
    def vector(self):
        if isinstance(self.args[1], Field.Access):
            return self.args[1].field
        else:
            return self.args[1]

    @property
    def scalar_index(self):
        return None if len(self.args) <= 2 else int(self.args[2])

    @property
    def dim(self):
        return self.scalar.spatial_dimensions

    def _latex(self, printer):
        name_suffix = f"_{self.scalar_index}" if self.scalar_index is not None else ""
        if isinstance(self.vector, Field):
            return r"\nabla \cdot(%s %s)" % (printer.doprint(sp.Symbol(self.vector.name)),
                                             printer.doprint(sp.Symbol(self.scalar.name + name_suffix)))
        else:
            args = [r"\partial_%d(%s %s)" % (i, printer.doprint(sp.Symbol(self.scalar.name + name_suffix)),
                                             printer.doprint(self.vector[i]))
                    for i in range(self.dim)]
            return " + ".join(args)

    # --- Interface for discretization strategy

    def velocity_field_at_offset(self, offset_dim, offset_value, index):
        v = self.vector
        if isinstance(v, Field):
            assert v.index_dimensions == 1
            return v.neighbor(offset_dim, offset_value)(index)
        else:
            return v[index]

    def advected_scalar_at_offset(self, offset_dim, offset_value):
        idx = 0 if self.scalar_index is None else int(self.scalar_index)
        return self.scalar.neighbor(offset_dim, offset_value)(idx)


class Diffusion(sp.Function):

    @property
    def scalar(self):
        return self.args[0].field

    @property
    def diffusion_coeff(self):
        if isinstance(self.args[1], Field.Access):
            return self.args[1].field
        else:
            return self.args[1]

    @property
    def scalar_index(self):
        return None if len(self.args) <= 2 else int(self.args[2])

    @property
    def dim(self):
        return self.scalar.spatial_dimensions

    def _latex(self, printer):
        name_suffix = f"_{self.scalar_index}" if self.scalar_index is not None else ""
        coeff = self.diffusion_coeff
        diff_coeff = sp.Symbol(coeff.name) if isinstance(coeff, Field) else coeff
        return r"div(%s \nabla %s)" % (printer.doprint(diff_coeff),
                                       printer.doprint(sp.Symbol(self.scalar.name + name_suffix)))

    # --- Interface for discretization strategy

    def diffusion_scalar_at_offset(self, offset_dim, offset_value):
        idx = 0 if self.scalar_index is None else self.scalar_index
        return self.scalar.neighbor(offset_dim, offset_value)(idx)

    def diffusion_coefficient_at_offset(self, offset_dim, offset_value):
        d = self.diffusion_coeff
        if isinstance(d, Field):
            return d.neighbor(offset_dim, offset_value)
        else:
            return d


class Transient(sp.Function):
    @property
    def scalar(self):
        if self.scalar_index is None:
            return self.args[0].field.center
        else:
            return self.args[0].field(self.scalar_index)

    @property
    def scalar_index(self):
        return None if len(self.args) <= 1 else int(self.args[1])

    def _latex(self, printer):
        name_suffix = f"_{self.scalar_index}" if self.scalar_index is not None else ""
        return r"\partial_t %s" % (printer.doprint(sp.Symbol(self.scalar.name + name_suffix)),)


# -------------------------------------------- Deprecated Functions ----------------------------------------------------


def grad(var, dim=3):
    r"""
    Gradients are represented as a special symbol:
    e.g. :math:`\nabla x = (x^{\Delta 0}, x^{\Delta 1}, x^{\Delta 2})`

    This function takes a symbol and creates the gradient symbols according to convention above

    Args:
        var: symbol to take the gradient of
        dim: dimension (length) of the gradient vector
    """
    if hasattr(var, "__getitem__"):
        return [[sp.Symbol("%s^Delta^%d" % (v.name, i)) for v in var] for i in range(dim)]
    else:
        return [sp.Symbol("%s^Delta^%d" % (var.name, i)) for i in range(dim)]


def discretize_center(term, symbols_to_field_dict, dx, dim=3):
    """
    Expects term that contains given symbols and gradient components of these symbols and replaces them
    by field accesses. Gradients are replaced by centralized approximations:
    ``(upper neighbor - lower neighbor ) / ( 2*dx)``

    Args:
        term: term where symbols and gradient(symbol) should be replaced
        symbols_to_field_dict: mapping of symbols to Field
        dx: width and height of one cell
        dim: dimension

    Example:
      >>> x = sp.Symbol("x")
      >>> grad_x = grad(x, dim=3)
      >>> term = x * grad_x[0]
      >>> term
      x*x^Delta^0
      >>> f = Field.create_generic('f', spatial_dimensions=3)
      >>> expected_output = f[0, 0, 0] * (-f[-1, 0, 0]/2 + f[1, 0, 0]/2)
      >>> sp.simplify(discretize_center(term, { x: f }, dx=1, dim=3) - expected_output)
      0
    """
    substitutions = {}
    for symbols, field in symbols_to_field_dict.items():
        if not hasattr(symbols, "__getitem__"):
            symbols = [symbols]
        g = grad(symbols, dim)
        substitutions.update({symbol: field(i) for i, symbol in enumerate(symbols)})
        for d in range(dim):
            up, down = __up_down_offsets(d, dim)
            substitutions.update({g[d][i]: (field[up](i) - field[down](i)) / dx / 2 for i in range(len(symbols))})
    return fast_subs(term, substitutions)


def discretize_staggered(term, symbols_to_field_dict, coordinate, coordinate_offset, dx, dim=3):
    """
    Expects term that contains given symbols and gradient components of these symbols and replaces them
    by field accesses. Gradients in coordinate direction  are replaced by staggered version at cell boundary.
    Symbols themselves and gradients in other directions are replaced by interpolated version at cell face.

    Args:
        term: input term where symbols and gradients are replaced
        symbols_to_field_dict: mapping of symbols to Field
        coordinate: id for coordinate (0 for x, 1 for y, ... ) defining cell boundary.
                    Only gradients in this direction are replaced e.g. if symbol^Delta^coordinate
        coordinate_offset: either +1 or -1 for upper or lower face in coordinate direction
        dx: width and height of one cell
        dim: dimension

    Examples:
      Discretizing at right/east face of cell i.e. coordinate=0, offset=1)
      >>> x, dx = sp.symbols("x dx")
      >>> grad_x = grad(x, dim=3)
      >>> term = x * grad_x[0]
      >>> term
      x*x^Delta^0
      >>> f = Field.create_generic('f', spatial_dimensions=3)
      >>> discretize_staggered(term, symbols_to_field_dict={ x: f}, dx=dx, coordinate=0, coordinate_offset=1, dim=3)
      (-f_C + f_E)*(f_C/2 + f_E/2)/dx
    """
    assert coordinate_offset == 1 or coordinate_offset == -1
    assert 0 <= coordinate < dim

    substitutions = {}
    for symbols, field in symbols_to_field_dict.items():
        if not hasattr(symbols, "__getitem__"):
            symbols = [symbols]

        offset = [0] * dim
        offset[coordinate] = coordinate_offset
        offset = np.array(offset, dtype=int)

        gradient = grad(symbols)[coordinate]
        substitutions.update({s: (field[offset](i) + field(i)) / 2 for i, s in enumerate(symbols)})
        substitutions.update({g: (field[offset](i) - field(i)) / dx * coordinate_offset
                              for i, g in enumerate(gradient)})
        for d in range(dim):
            if d == coordinate:
                continue
            up, down = __up_down_offsets(d, dim)
            for i, s in enumerate(symbols):
                center_grad = (field[up](i) - field[down](i)) / (2 * dx)
                neighbor_grad = (field[up + offset](i) - field[down + offset](i)) / (2 * dx)
                substitutions[grad(s)[d]] = (center_grad + neighbor_grad) / 2

    return fast_subs(term, substitutions)


def discretize_divergence(vector_term, symbols_to_field_dict, dx):
    """
    Computes discrete divergence of symbolic vector

    Args:
        vector_term: sequence of terms, interpreted as vector
        symbols_to_field_dict: mapping of symbols to Field
        dx: length of a cell

    Examples:
        Laplace stencil
        >>> x, dx = sp.symbols("x dx")
        >>> grad_x = grad(x, dim=3)
        >>> f = Field.create_generic('f', spatial_dimensions=3)
        >>> expected_output = (f[-1, 0, 0] + f[0, -1, 0] + f[0, 0, -1] -
        ... 6*f[0, 0, 0] + f[0, 0, 1] + f[0, 1, 0] + f[1, 0, 0])/dx**2
        >>> sp.simplify(discretize_divergence(grad_x, {x : f}, dx) - expected_output)
        0
    """
    dim = len(vector_term)
    result = 0
    for d in range(dim):
        for offset in [-1, 1]:
            result += offset * discretize_staggered(vector_term[d], symbols_to_field_dict, d, offset, dx, dim)
    return result / dx


def __up_down_offsets(d, dim):
    coord = [0] * dim
    coord[d] = 1
    up = np.array(coord, dtype=int)
    coord[d] = -1
    down = np.array(coord, dtype=int)
    return up, down
