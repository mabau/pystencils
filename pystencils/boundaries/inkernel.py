import sympy as sp
from pystencils import Field, TypedSymbol
from pystencils.integer_functions import bitwise_and
from pystencils.boundaries.boundaryhandling import DEFAULT_FLAG_TYPE
from pystencils.data_types import create_type


def add_neumann_boundary(eqs, fields, flag_field, boundary_flag="neumann_flag", inverse_flag=False):
    """
    Replaces all neighbor accesses by flag field guarded accesses.
    If flag in neighboring cell is set, the center value is used instead
    :param eqs: list of equations containing field accesses to direct neighbors
    :param fields: fields for which the Neumann boundary should be applied
    :param flag_field: integer field marking boundary cells
    :param boundary_flag: if flag field has value 'boundary_flag' (no bit operations yet)
                          the cell is assumed to be boundary
    :param inverse_flag: if true, boundary cells are where flag field has not the value of boundary_flag
    :return: list of equations with guarded field accesses
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
