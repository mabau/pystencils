from functools import reduce
from typing import Any, cast

from pymbolic.primitives import Variable
from pymbolic.mapper.dependency import DependencyMapper

from .kernelfunction import PsKernelFunction
from .nodes import PsAstNode, PsExpression, PsAssignment, PsDeclaration, PsLoop, PsBlock
from ..typed_expressions import PsTypedVariable
from ..exceptions import PsMalformedAstException, PsInternalCompilerError


class UndefinedVariablesCollector:
    """Collector for undefined variables.

    This class implements an AST visitor that collects all `PsTypedVariable`s that have been used
    in the AST without being defined prior to their usage.
    """

    def __init__(self) -> None:
        self._pb_dep_mapper = DependencyMapper(
            include_subscripts=False,
            include_lookups=False,
            include_calls=False,
            include_cses=False,
        )

    def collect(self, node: PsAstNode) -> set[PsTypedVariable]:
        """Returns all `PsTypedVariable`s that occur in the given AST without being defined prior to their usage."""

        match node:
            case PsKernelFunction(block):
                return self.collect(block)

            case PsExpression(expr):
                variables: set[Variable] = self._pb_dep_mapper(expr)

                for var in variables:
                    if not isinstance(var, PsTypedVariable):
                        raise PsMalformedAstException(
                            f"Non-typed variable {var} encountered"
                        )

                return cast(set[PsTypedVariable], variables)

            case PsAssignment(lhs, rhs):
                return self.collect(lhs) | self.collect(rhs)

            case PsBlock(statements):
                undefined_vars = set()
                for stmt in statements[::-1]:
                    undefined_vars -= self.declared_variables(stmt)
                    undefined_vars |= self.collect(stmt)

                return undefined_vars

            case PsLoop(ctr, start, stop, step, body):
                undefined_vars = (
                    self.collect(start)
                    | self.collect(stop)
                    | self.collect(step)
                    | self.collect(body)
                )
                undefined_vars.remove(ctr.symbol)
                return undefined_vars

            case unknown:
                raise PsInternalCompilerError(
                    f"Don't know how to collect undefined variables from {unknown}"
                )

    def declared_variables(self, node: PsAstNode) -> set[PsTypedVariable]:
        """Returns the set of variables declared by the given node which are visible in the enclosing scope."""

        match node:
            case PsDeclaration(lhs, _):
                return {lhs.symbol}

            case PsAssignment() | PsExpression() | PsLoop() | PsBlock():
                return set()

            case unknown:
                raise PsInternalCompilerError(
                    f"Don't know how to collect declared variables from {unknown}"
                )
