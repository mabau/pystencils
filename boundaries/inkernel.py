import sympy as sp
from pystencils import Field, TypedSymbol
from pystencils.bitoperations import bitwiseAnd
from pystencils.boundaries.boundaryhandling import FlagInterface
from pystencils.data_types import createType


def addNeumannBoundary(eqs, fields, flagField, boundaryFlag="neumannFlag", inverseFlag=False):
    """
    Replaces all neighbor accesses by flag field guarded accesses.
    If flag in neighboring cell is set, the center value is used instead
    :param eqs: list of equations containing field accesses to direct neighbors
    :param fields: fields for which the Neumann boundary should be applied
    :param flagField: integer field marking boundary cells
    :param boundaryFlag: if flag field has value 'boundaryFlag' (no bitoperations yet) the cell is assumed to be boundary
    :param inverseFlag: if true, boundary cells are where flagfield has not the value of boundaryFlag
    :return: list of equations with guarded field accesses
    """
    if not hasattr(fields, "__len__"):
        fields = [fields]
    fields = set(fields)

    if type(boundaryFlag) is str:
        boundaryFlag = TypedSymbol(boundaryFlag, dtype=createType(FlagInterface.FLAG_DTYPE))

    substitutions = {}
    for eq in eqs:
        for fa in eq.atoms(Field.Access):
            if fa.field not in fields:
                continue
            if not all(offset in (-1, 0, 1) for offset in fa.offsets):
                raise ValueError("Works only for single neighborhood stencils")
            if all(offset == 0 for offset in fa.offsets):
                continue

            if inverseFlag:
                condition = sp.Eq(bitwiseAnd(flagField[tuple(fa.offsets)], boundaryFlag), 0)
            else:
                condition = sp.Ne(bitwiseAnd(flagField[tuple(fa.offsets)], boundaryFlag), 0)

            center = fa.field(*fa.index)
            substitutions[fa] = sp.Piecewise((center, condition), (fa, True))
    return [eq.subs(substitutions) for eq in eqs]
