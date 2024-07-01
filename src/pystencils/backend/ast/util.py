from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .astnode import PsAstNode


def failing_cast(target: type | tuple[type, ...], obj: Any) -> Any:
    if not isinstance(obj, target):
        raise TypeError(f"Casting {obj} to {target} failed.")
    return obj


class AstEqWrapper:
    """Wrapper around AST nodes that computes a hash from the AST's textual representation
    and maps the `__eq__` method onto `structurally_equal`.

    Useful in dictionaries when the goal is to keep track of subtrees according to their
    structure, e.g. in elimination of constants or common subexpressions.
    """

    def __init__(self, node: PsAstNode):
        self._node = node

    @property
    def n(self):
        return self._node

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AstEqWrapper):
            return False

        return self._node.structurally_equal(other._node)

    def __hash__(self) -> int:
        #   TODO: consider replacing this with smth. more performant
        #   TODO: Check that repr is implemented by all AST nodes
        return hash(repr(self._node))


def c_intdiv(num, denom):
    """C-style integer division"""
    return int(num / denom)


def c_rem(num, denom):
    """C-style integer remainder"""
    div = c_intdiv(num, denom)
    return num - div * denom
