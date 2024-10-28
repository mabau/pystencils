from __future__ import annotations
from typing import Any, TYPE_CHECKING, cast

from ..exceptions import PsInternalCompilerError
from ..memory import PsSymbol
from ..memory import PsBuffer
from ...types import PsDereferencableType


if TYPE_CHECKING:
    from .astnode import PsAstNode
    from .expressions import PsExpression


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


def determine_memory_object(
    expr: PsExpression,
) -> tuple[PsSymbol | PsBuffer | None, bool]:
    """Return the memory object accessed by the given expression, together with its constness

    Returns:
        Tuple ``(mem_obj, const)`` identifying the memory object accessed by the given expression,
        as well as its constness
    """
    from pystencils.backend.ast.expressions import (
        PsSubscript,
        PsLookup,
        PsSymbolExpr,
        PsMemAcc,
        PsBufferAcc,
    )

    while isinstance(expr, (PsSubscript, PsLookup)):
        match expr:
            case PsSubscript(arr, _):
                expr = arr
            case PsLookup(record, _):
                expr = record

    match expr:
        case PsSymbolExpr(symb):
            return symb, symb.get_dtype().const
        case PsMemAcc(ptr, _):
            return None, cast(PsDereferencableType, ptr.get_dtype()).base_type.const
        case PsBufferAcc(ptr, _):
            return (
                expr.buffer,
                cast(PsDereferencableType, ptr.get_dtype()).base_type.const,
            )
        case _:
            raise PsInternalCompilerError(
                "The given expression is a transient and does not refer to a memory object"
            )
