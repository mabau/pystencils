import sympy as sp

from pystencils.boundaries.boundaryhandling import DEFAULT_FLAG_TYPE
from pystencils.typing import TypedSymbol, create_type
from pystencils.field import Field
from pystencils.integer_functions import bitwise_and


def add_neumann_boundary(eqs, fields, flag_field, boundary_flag="neumann_flag", inverse_flag=False):
    """
    Replaces all neighbor accesses by flag field guarded accesses.
    If flag in neighboring cell is set, the center value is used instead

    Args:
        eqs: list of equations containing field accesses to direct neighbors
        fields: fields for which the Neumann boundary should be applied
        flag_field: integer field marking boundary cells
        boundary_flag: if flag field has value 'boundary_flag' (no bit operations yet)
                       the cell is assumed to be boundary
        inverse_flag: if true, boundary cells are where flag field has not the value of boundary_flag

    Returns:
        list of equations with guarded field accesses
    """
    if not hasattr(fields, "__len__"):
        fields = [fields]
    fields = set(fields)

    if type(boundary_flag) is str:
        boundary_flag = TypedSymbol(boundary_flag, dtype=create_type(DEFAULT_FLAG_TYPE))

    substitutions = {}
    for eq in eqs:
        for fa in eq.atoms(Field.Access):
            if fa.field not in fields:
                continue
            if not all(offset in (-1, 0, 1) for offset in fa.offsets):
                raise ValueError("Works only for single neighborhood stencils")
            if all(offset == 0 for offset in fa.offsets):
                continue

            if inverse_flag:
                condition = sp.Eq(bitwise_and(flag_field[tuple(fa.offsets)], boundary_flag), 0)
            else:
                condition = sp.Ne(bitwise_and(flag_field[tuple(fa.offsets)], boundary_flag), 0)

            center = fa.field(*fa.index)
            substitutions[fa] = sp.Piecewise((center, condition), (fa, True))
    return [eq.subs(substitutions) for eq in eqs]
