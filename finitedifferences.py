import numpy as np
import sympy as sp
from pystencils.field import Field
from pystencils.transformations import fastSubs


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
    by field accesses. Gradients are replaced by centralized approximations:
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
        (f_B - 6*f_C + f_E + f_N + f_S + f_T + f_W)/dx**2
    """
    dim = len(vectorTerm)
    result = 0
    for d in range(dim):
        for offset in [-1, 1]:
            result += offset * discretizeStaggered(vectorTerm[d], symbolsToFieldDict, d, offset, dx, dim)
    return result / dx


def __upDownOffsets(d, dim):
    coord = [0] * dim
    coord[d] = 1
    up = np.array(coord, dtype=np.int)
    coord[d] = -1
    down = np.array(coord, dtype=np.int)
    return up, down


# --------------------------------------- Advection Diffusion ----------------------------------------------------------


class Advection(sp.Function):
    """Advection term, create with advection(scalarField, vectorField)"""

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
    def scalarIndex(self):
        return None if len(self.args) <= 2 else int(self.args[2])

    @property
    def dim(self):
        return self.scalar.spatialDimensions

    def _latex(self, printer):
        nameSuffix = "_%s" % self.scalarIndex if self.scalarIndex is not None else ""
        if isinstance(self.vector, Field):
            return r"\nabla \cdot(%s %s)" % (printer.doprint(sp.Symbol(self.vector.name)),
                                             printer.doprint(sp.Symbol(self.scalar.name+nameSuffix)))
        else:
            args = [r"\partial_%d(%s %s)" % (i, printer.doprint(sp.Symbol(self.scalar.name+nameSuffix)),
                                             printer.doprint(self.vector[i]))
                    for i in range(self.dim)]
            return " + ".join(args)


def advection(scalar, vector, idx=None):
    assert isinstance(scalar, Field), "Advected scalar has to be a pystencils.Field"

    args = [scalar.center, vector if not isinstance(vector, Field) else vector.center]
    if idx is not None:
        args.append(idx)
    return Advection(*args)


class Diffusion(sp.Function):

    @property
    def scalar(self):
        return self.args[0].field

    @property
    def diffusionCoeff(self):
        if isinstance(self.args[1], Field.Access):
            return self.args[1].field
        else:
            return self.args[1]

    @property
    def scalarIndex(self):
        return None if len(self.args) <= 2 else int(self.args[2])

    @property
    def dim(self):
        return self.scalar.spatialDimensions

    def _latex(self, printer):
        nameSuffix = "_%s" % self.scalarIndex if self.scalarIndex is not None else ""
        diffCoeff = sp.Symbol(self.diffusionCoeff.name) if isinstance(self.diffusionCoeff, Field) else self.diffusionCoeff
        return r"div(%s \nabla %s)" % (printer.doprint(diffCoeff),
                                       printer.doprint(sp.Symbol(self.scalar.name+nameSuffix)))


def diffusion(scalar, diffusionCoeff, idx=None):
    assert isinstance(scalar, Field), "Advected scalar has to be a pystencils.Field"
    args = [scalar.center, diffusionCoeff if not isinstance(diffusionCoeff, Field) else diffusionCoeff.center]
    if idx is not None:
        args.append(idx)
    return Diffusion(*args)


class Transient(sp.Function):
    @property
    def scalar(self):
        return self.args[0].field

    @property
    def scalarIndex(self):
        return None if len(self.args) <= 1 else int(self.args[1])

    def _latex(self, printer):
        nameSuffix = "_%s" % self.scalarIndex if self.scalarIndex is not None else ""
        return r"\partial_t %s" % (printer.doprint(sp.Symbol(self.scalar.name+nameSuffix)),)


def transient(scalar, idx=None):
    args = [scalar.center]
    if idx is not None:
        args.append(idx)
    return Transient(*args)


class Discretization2ndOrder:
    def __init__(self, dx=sp.Symbol("dx"), dt=sp.Symbol("dt")):
        self.dx = dx
        self.dt = dt

    def _discretize_diffusion(self, expr):
        result = 0
        scalar = expr.scalar
        for c in range(expr.dim):
            if isinstance(expr.diffusionCoeff, Field):
                firstDiffs = [offset *
                              (scalar.neighbor(c, offset) * expr.diffusionCoeff.neighbor(c, offset) -
                               scalar.center * expr.diffusionCoeff.center())
                              for offset in [-1, 1]]
            else:
                firstDiffs = [offset *
                              (scalar.neighbor(c, offset) * expr.diffusionCoeff -
                               scalar.center * expr.diffusionCoeff)
                              for offset in [-1, 1]]
            result += firstDiffs[1] - firstDiffs[0]
        return result / (self.dx**2)

    def _discretize_advection(self, expr):
        idx = 0 if expr.scalarIndex is None else int(expr.scalarIndex)
        result = 0
        for c in range(expr.dim):
            if isinstance(expr.vector, Field):
                assert expr.vector.indexDimensions == 1
                interpolated = [(expr.scalar.neighbor(c, offset)(idx) * expr.vector.neighbor(c, offset)(c) +
                                 expr.scalar.neighbor(c, 0)(idx) * expr.vector.neighbor(c, 0)(c)) / 2
                                for offset in [-1, 1]]
            else:
                interpolated = [(expr.scalar.neighbor(c, offset)(idx) * expr.vector(c) -
                                 expr.scalar.neighbor(c, 0)(idx) * expr.vector(c)) / 2
                                for offset in [-1, 1]]
            result += interpolated[1] - interpolated[0]
        return result / self.dx

    def _discretizeSpatial(self, e):
        if isinstance(e, Diffusion):
            return self._discretize_diffusion(e)
        elif isinstance(e, Advection):
            return self._discretize_advection(e)
        else:
            newArgs = [self._discretizeSpatial(a) for a in e.args]
            return e.func(*newArgs) if newArgs else e

    def __call__(self, expr):
        transientTerms = expr.atoms(Transient)
        if len(transientTerms) == 0:
            return self._discretizeSpatial(expr)
        elif len(transientTerms) == 1:
            transientTerm = transientTerms.pop()
            solveResult = sp.solve(expr, transientTerm)
            if len(solveResult) != 1:
                raise ValueError("Could not solve for transient term" + str(solveResult))
            rhs = solveResult.pop()
            idx = 0 if transientTerm.scalarIndex is None else transientTerm.scalarIndex
            # explicit euler
            return transientTerm.scalar(idx) + self.dt * self._discretizeSpatial(rhs)
        else:
            raise NotImplementedError("Cannot discretize expression with more than one transient term")


