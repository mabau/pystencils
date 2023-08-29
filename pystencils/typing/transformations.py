from typing import List

from pystencils.astnodes import Node
from pystencils.config import CreateKernelConfig
from pystencils.typing.leaf_typing import TypeAdder


def add_types(node_list: List[Node], config: CreateKernelConfig):
    """Traverses AST and replaces every :class:`sympy.Symbol` by a :class:`pystencils.typedsymbol.TypedSymbol`.
    The AST needs to be a pystencils AST. Thus, in the list of nodes every entry must be inherited from
    `pystencils.astnodes.Node`

    Additionally returns sets of all fields which are read/written

    Args:
        node_list: List of pystencils Nodes.
        config: CreateKernelConfig

    Returns:
        ``typed_equations`` list of equations where symbols have been replaced by typed symbols
    """

    check = TypeAdder(type_for_symbol=config.data_type,
                      default_number_float=config.default_number_float,
                      default_number_int=config.default_number_int)

    return check.visit(node_list)
