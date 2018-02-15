import sympy as sp
from pystencils.boundaries.boundaryhandling import BoundaryOffsetInfo


class Boundary(object):
    """Base class all boundaries should derive from"""

    def __init__(self, name=None):
        self._name = name

    def __call__(self, field, directionSymbol, indexField):
        """
        This function defines the boundary behavior and must therefore be implemented by all boundaries.
        Here the boundary is defined as a list of sympy equations, from which a boundary kernel is generated.
        :param field: pystencils field where boundary condition should be applied.
                     The current cell is cell next to the boundary, which is influenced by the boundary
                     cell i.e. has a link from the boundary cell to itself.
        :param directionSymbol: a sympy symbol that can be used as index to the pdfField. It describes
                                the direction pointing from the fluid to the boundary cell
        :param indexField: the boundary index field that can be used to retrieve and update boundary data
        :return: list of sympy equations
        """
        raise NotImplementedError("Boundary class has to overwrite __call__")

    @property
    def additionalData(self):
        """Return a list of (name, type) tuples for additional data items required in this boundary
        These data items can either be initialized in separate kernel see additionalDataKernelInit or by
        Python callbacks - see additionalDataCallback """
        return []

    @property
    def additionalDataInitCallback(self):
        """Return a callback function called with a boundary data setter object and returning a dict of
        data-name to data for each element that should be initialized"""
        return None

    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return type(self).__name__

    @name.setter
    def name(self, newValue):
        self._name = newValue


class Neumann(Boundary):
    def __call__(self, field, directionSymbol, **kwargs):

        neighbor = BoundaryOffsetInfo.offsetFromDir(directionSymbol, field.spatialDimensions)
        if field.indexDimensions == 0:
            return [sp.Eq(field[neighbor], field.center)]
        else:
            from itertools import product
            if not field.hasFixedIndexShape:
                raise NotImplementedError("Neumann boundary works only for fields with fixed index shape")
            indexIter = product(*(range(i) for i in field.indexShape))
            return [sp.Eq(field[neighbor](idx), field(idx)) for idx in indexIter]

    def __hash__(self):
        # All boundaries of these class behave equal -> should also be equal
        return hash("Neumann")

    def __eq__(self, other):
        return type(other) == Neumann
