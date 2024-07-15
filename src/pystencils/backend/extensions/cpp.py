from __future__ import annotations
from typing import Iterable, cast

from pystencils.backend.ast.astnode import PsAstNode

from ..ast.expressions import PsExpression
from .foreign_ast import PsForeignExpression
from ...types import PsType


class CppMethodCall(PsForeignExpression):
    """C++ method call on an expression."""

    def __init__(
        self, obj: PsExpression, method: str, return_type: PsType, args: Iterable = ()
    ):
        self._method = method
        self._return_type = return_type
        children = [obj] + list(args)
        super().__init__(children, return_type)

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, CppMethodCall):
            return False

        return super().structurally_equal(other) and self._method == other._method

    def clone(self) -> CppMethodCall:
        return CppMethodCall(
            cast(PsExpression, self.children[0]),
            self._method,
            self._return_type,
            self.children[1:],
        )

    def get_code(self, children_code: Iterable[str]) -> str:
        cs = list(children_code)
        obj_code = cs[0]
        args_code = cs[1:]
        args = ", ".join(args_code)
        return f"({obj_code}).{self._method}({args})"
