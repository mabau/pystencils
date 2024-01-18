from __future__ import annotations

from typing import TypeVar

import pymbolic.primitives as pb
from pymbolic.mapper import Mapper

from .context import KernelCreationContext
from ..types import PsAbstractType
from ..typed_expressions import PsTypedVariable
from ..ast import PsAstNode, PsExpression, PsAssignment


class TypificationException(Exception):
    """Indicates a fatal error during typification."""


NodeT = TypeVar("NodeT", bound=PsAstNode)


class Typifier(Mapper):
    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

    def __call__(self, node: NodeT) -> NodeT:
        match node:
            case PsExpression(expr):
                node.expression, _ = self.rec(expr)

            case PsAssignment(lhs, rhs):
                lhs, lhs_dtype = self.rec(lhs)
                rhs, rhs_dtype = self.rec(rhs)
                if lhs_dtype != rhs_dtype:
                    #   todo: (optional) automatic cast insertion?
                    raise TypificationException(
                        "Mismatched types in assignment: \n"
                        f"    {lhs} <- {rhs}\n"
                        f"    dtype(lhs) = {lhs_dtype}\n"
                        f"    dtype(rhs) = {rhs_dtype}\n"
                    )
                node.lhs = lhs
                node.rhs = rhs

            case unknown:
                raise NotImplementedError(f"Don't know how to typify {unknown}")
            
        return node

    def map_variable(self, var: pb.Variable) -> tuple[pb.Expression, PsAbstractType]:
        dtype = NotImplemented  # determine variable type
        return PsTypedVariable(var.name, dtype), dtype
