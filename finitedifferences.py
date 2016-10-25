import collections
import sympy as sp
from lbmpy.generator import Field


def __upDownTuples(d, dim):
    coord = [0] * dim
    coord[d] = 1
    up = tuple(coord)
    coord[d] = -1
    down = tuple(coord)
    return up, down


def grad(var, dim=3):
    """Gradients are represented as a special symbol:
    e.g. :math:`\nabla x = (x^\Delta^0, x^\Delta^1, x^\Delta^2)`
    This function takes a symbol and creates the gradient symbols according to convention above
    :param var: symbol to take the gradient of
    :param dim: dimension (length) of the gradient vector
    """
    if hasattr(var, "__getitem__"):
        return [[sp.Symbol("%s^Delta^%d" % (v.name, i)) for v in var] for i in range(dim)]
    else:
        return [sp.Symbol("%s^Delta^%d" % (var.name, i)) for i in range(dim)]


def discretizeCenter(term, symbols, field, dx, dim=3):
    """
    Expects term that contains given symbols and gradient components of these symbols and replaces them
    by field accesses. Gradients are replaced centralized approximations: (upper neighbor - lower neighbor ) / ( 2*dx).
    :param term: term where symbols and gradient(symbol) should be replaced
    :param symbols: these symbols and their gradients are replaced by field accesses
    :param field: field containing the discrete values for symbols
    :param dx: width and height of one cell
    :param dim: dimension

    Example:
      >>> x = sp.Symbol("x")
      >>> gradx = grad(x, dim=3)
      >>> term = x * gradx[0]
      >>> term
      x*x^Delta^0
      >>> f = Field.createGeneric('f', spatialDimensions=3)
      >>> discretizeCenter(term, symbols=x, field=f, dx=1, dim=3)
      f_C*(f_E/2 - f_W/2)
    """
    if not hasattr(symbols, "__getitem__"):
        symbols = [symbols]
    g = grad(symbols, dim)
    substitutions = {symbol: field(i) for i, symbol in enumerate(symbols)}
    for d in range(dim):
        up, down = __upDownTuples(d, dim)
        substitutions.update({g[d][i]: (field[up](i) - field[down](i)) / dx / 2 for i in range(len(symbols))})
    return term.subs(substitutions)


def discretizeStaggered(term, symbols, field, coordinate, offset, dx, dim=3):
    """
    Expects term that contains given symbols and gradient components of these symbols and replaces them
    by field accesses. Gradients in coordinate direction  are replaced by staggered version at cell boundary.
    Symbols themselves are replaced by interpolated version at boundary.

    :param term: input term where symbols and gradients are replaced
    :param symbols: these symbols and their gradient in coordinate direction is replaced
    :param field: field containing the discrete values for symbols
    :param coordinate: id for coordinate (0 for x, 1 for y, ... ) defining cell boundary.
                       Only gradients in this direction are replaced e.g. if symbol^Delta^coordinate
    :param offset: either +1 or -1 for upper or lower face in coordinate direction
    :param dx: width and height of one cell
    :param dim: dimension

    Example: Discretizing at right/east face of cell i.e. coordinate=0, offset=1)
      >>> x, dx = sp.symbols("x dx")
      >>> gradx = grad(x, dim=3)
      >>> term = x * gradx[0]
      >>> term
      x*x^Delta^0
      >>> f = Field.createGeneric('f', spatialDimensions=3)
      >>> discretizeStaggered(term, symbols=x, field=f, dx=dx, coordinate=0, offset=1, dim=3)
      (-f_C + f_E)*(f_C/2 + f_E/2)/dx
    """
    assert offset == 1 or offset == -1
    assert 0 <= coordinate < dim
    if not isinstance(symbols, collections.Sequence):
        symbols = [symbols]

    offsetTuple = [0] * dim
    offsetTuple[coordinate] = offset
    offsetTuple = tuple(offsetTuple)

    gradient = grad(symbols)[coordinate]
    substitutions = {s: (field[offsetTuple](i) + field(i)) / 2 for i, s in enumerate(symbols)}
    substitutions.update({g: (field[offsetTuple](i) - field(i)) / dx * offset for i, g in enumerate(gradient)})
    return term.subs(substitutions)


def discretizeDivergence(vectorTerm, symbols, field, dx):
    """
    Computes discrete divergence of symbolic vector
    :param vectorTerm: sequence of terms, interpreted as vector
    :param symbols: these symbols and their gradient in coordinate direction is replaced
    :param field: field containing the discrete values for symbols

    Example: Laplace stencil
      >>> x, dx = sp.symbols("x dx")
      >>> gradx = grad(x, dim=3)
      >>> f = Field.createGeneric('f', spatialDimensions=3)
      >>> sp.simplify(discretizeDivergence(gradx, x, f, dx))
      (f_B - 6*f_C + f_E + f_N + f_S + f_T + f_W)/dx
    """
    dim = len(vectorTerm)
    result = 0
    for d in range(dim):
        for offset in [-1, 1]:
            result += offset * discretizeStaggered(vectorTerm[d], symbols, field, d, offset, dx, dim)
    return result
