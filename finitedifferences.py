import numpy as np
import sympy as sp
from pystencils.field import Field


def grad(var, dim=3):
    r"""
    Gradients are represented as a special symbol:
    e.g. :math:`\nabla x = (x^{\Delta 0}, x^{\Delta 1}, x^{\Delta 2})`

    This function takes a symbol and creates the gradient symbols according to convention above

    :param var: symbol to take the gradient of
    :param dim: dimension (length) of the gradient vector
    """
    if hasattr(var, "__getitem__"):
        return [[sp.Symbol("%s^Delta^%d" % (v.name, i)) for v in var] for i in range(dim)]
    else:
        return [sp.Symbol("%s^Delta^%d" % (var.name, i)) for i in range(dim)]


def discretizeCenter(term, symbolsToFieldDict, dx, dim=3):
    """
    Expects term that contains given symbols and gradient components of these symbols and replaces them
    by field accesses. Gradients are replaced centralized approximations:
    ``(upper neighbor - lower neighbor ) / ( 2*dx)``
    :param term: term where symbols and gradient(symbol) should be replaced
    :param symbolsToFieldDict: mapping of symbols to Field
    :param dx: width and height of one cell
    :param dim: dimension

    Example:
      >>> x = sp.Symbol("x")
      >>> gradx = grad(x, dim=3)
      >>> term = x * gradx[0]
      >>> term
      x*x^Delta^0
      >>> f = Field.createGeneric('f', spatialDimensions=3)
      >>> discretizeCenter(term, { x: f }, dx=1, dim=3)
      f_C*(f_E/2 - f_W/2)
    """
    substitutions = {}
    for symbols, field in symbolsToFieldDict.items():
        if not hasattr(symbols, "__getitem__"):
            symbols = [symbols]
        g = grad(symbols, dim)
        substitutions.update({symbol: field(i) for i, symbol in enumerate(symbols)})
        for d in range(dim):
            up, down = __upDownOffsets(d, dim)
            substitutions.update({g[d][i]: (field[up](i) - field[down](i)) / dx / 2 for i in range(len(symbols))})
    return term.subs(substitutions)


def discretizeStaggered(term, symbolsToFieldDict, coordinate, coordinateOffset, dx, dim=3):
    """
    Expects term that contains given symbols and gradient components of these symbols and replaces them
    by field accesses. Gradients in coordinate direction  are replaced by staggered version at cell boundary.
    Symbols themselves and gradients in other directions are replaced by interpolated version at cell face.

    :param term: input term where symbols and gradients are replaced
    :param symbolsToFieldDict: mapping of symbols to Field
    :param coordinate: id for coordinate (0 for x, 1 for y, ... ) defining cell boundary.
                       Only gradients in this direction are replaced e.g. if symbol^Delta^coordinate
    :param coordinateOffset: either +1 or -1 for upper or lower face in coordinate direction
    :param dx: width and height of one cell
    :param dim: dimension

    Example: Discretizing at right/east face of cell i.e. coordinate=0, offset=1)
      >>> x, dx = sp.symbols("x dx")
      >>> gradx = grad(x, dim=3)
      >>> term = x * gradx[0]
      >>> term
      x*x^Delta^0
      >>> f = Field.createGeneric('f', spatialDimensions=3)
      >>> discretizeStaggered(term, symbolsToFieldDict={ x: f}, dx=dx, coordinate=0, coordinateOffset=1, dim=3)
      (-f_C + f_E)*(f_C/2 + f_E/2)/dx
    """
    assert coordinateOffset == 1 or coordinateOffset == -1
    assert 0 <= coordinate < dim

    substitutions = {}
    for symbols, field in symbolsToFieldDict.items():
        if not hasattr(symbols, "__getitem__"):
            symbols = [symbols]

        offset = [0] * dim
        offset[coordinate] = coordinateOffset
        offset = np.array(offset, dtype=np.int)

        gradient = grad(symbols)[coordinate]
        substitutions.update({s: (field[offset](i) + field(i)) / 2 for i, s in enumerate(symbols)})
        substitutions.update({g: (field[offset](i) - field(i)) / dx * coordinateOffset for i, g in enumerate(gradient)})
        for d in range(dim):
            if d == coordinate:
                continue
            up, down = __upDownOffsets(d, dim)
            for i, s in enumerate(symbols):
                centerGrad = (field[up](i) - field[down](i)) / (2 * dx)
                neighborGrad = (field[up+offset](i) - field[down+offset](i)) / (2 * dx)
                substitutions[grad(s)[d]] = (centerGrad + neighborGrad) / 2

    return fastSubs(term, substitutions)


def discretizeDivergence(vectorTerm, symbolsToFieldDict, dx):
    """
    Computes discrete divergence of symbolic vector
    :param vectorTerm: sequence of terms, interpreted as vector
    :param symbolsToFieldDict: mapping of symbols to Field
    :param dx: length of a cell

    Example: Laplace stencil
        >>> x, dx = sp.symbols("x dx")
        >>> gradX = grad(x, dim=3)
        >>> f = Field.createGeneric('f', spatialDimensions=3)
        >>> sp.simplify(discretizeDivergence(gradX, {x : f}, dx))
        (f_B - 6*f_C + f_E + f_N + f_S + f_T + f_W)/dx
    """
    dim = len(vectorTerm)
    result = 0
    for d in range(dim):
        for offset in [-1, 1]:
            result += offset * discretizeStaggered(vectorTerm[d], symbolsToFieldDict, d, offset, dx, dim)
    return result


def __upDownOffsets(d, dim):
    coord = [0] * dim
    coord[d] = 1
    up = np.array(coord, dtype=np.int)
    coord[d] = -1
    down = np.array(coord, dtype=np.int)
    return up, down


def fastSubs(term, subsDict):
    """Similar to sympy subs function.
    This version is much faster for big substitution dictionaries than sympy version"""
    def visit(expr):
        if expr in subsDict:
            return subsDict[expr]
        paramList = [visit(a) for a in expr.args]
        return expr if not paramList else expr.func(*paramList)
    return visit(term)
