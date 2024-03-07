from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .astnode import PsAstNode


def failing_cast(target: type | tuple[type, ...], obj: Any) -> Any:
    if not isinstance(obj, target):
        raise TypeError(f"Casting {obj} to {target} failed.")
    return obj


class EqWrapper:
    """Wrapper around AST nodes that maps the `__eq__` method onto `structurally_equal`.

    Useful in dictionaries when the goal is to keep track of subtrees according to their
    structure, e.g. in elimination of constants or common subexpressions.
    """

    def __init__(self, node: PsAstNode):
        self._node = node

    @property
    def n(self):
        return self._node

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EqWrapper):
            return False

        return self._node.structurally_equal(other._node)

    def __hash__(self) -> int:
        return hash(self._node)
