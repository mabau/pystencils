from typing import List

from pystencils.config import CreateKernelConfig
from pystencils.typing.leaf_typing import TypeAdder
from sympy.codegen import Assignment


def add_types(eqs: List[Assignment], config: CreateKernelConfig):
    """Traverses AST and replaces every :class:`sympy.Symbol` by a :class:`pystencils.typedsymbol.TypedSymbol`.

    Additionally returns sets of all fields which are read/written

    Args:
        eqs: list of equations
        config: CreateKernelConfig

    Returns:
        ``typed_equations`` list of equations where symbols have been replaced by typed symbols
    """

    check = TypeAdder(type_for_symbol=config.data_type,
                      default_number_float=config.default_number_float,
                      default_number_int=config.default_number_int)

    return check.visit(eqs)
