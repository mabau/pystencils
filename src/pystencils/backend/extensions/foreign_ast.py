from __future__ import annotations
from typing import Iterable
from abc import ABC, abstractmethod

from pystencils.backend.ast.astnode import PsAstNode

from ..ast.expressions import PsExpression
from ..ast.util import failing_cast
from ...types import PsType


class PsForeignExpression(PsExpression, ABC):
    """Base class for foreign expressions.

    Foreign expressions are expressions whose properties are not modelled by the pystencils AST,
    and which pystencils therefore does not understand.

    There are many situations where non-supported expressions are needed;
    the most common use case is C++ syntax.
    Support for foreign expressions by the code generator is therefore very limited;
    as a rule of thumb, only printing is supported.
    Type checking and most transformations will fail when encountering a `PsForeignExpression`.
    """

    __match_args__ = ("children",)

    def __init__(self, children: Iterable[PsExpression], dtype: PsType | None = None):
        self._children = list(children)
        super().__init__(dtype)

    @abstractmethod
    def get_code(self, children_code: Iterable[str]) -> str:
        """Print this expression, with the given code for each of its children."""
        pass

    def get_children(self) -> tuple[PsAstNode, ...]:
        return tuple(self._children)

    def set_child(self, idx: int, c: PsAstNode):
        self._children[idx] = failing_cast(PsExpression, c)

    def __repr__(self) -> str:
        return f"{type(self)}({self._children})"
