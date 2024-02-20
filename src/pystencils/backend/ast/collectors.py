from typing import cast, Any

from functools import reduce

from pymbolic.primitives import Variable
from pymbolic.mapper import Collector
from pymbolic.mapper.dependency import DependencyMapper

from .kernelfunction import PsKernelFunction
from .nodes import PsAstNode, PsExpression, PsAssignment, PsDeclaration, PsLoop, PsBlock
from ..typed_expressions import PsTypedVariable, PsTypedConstant
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

    def __call__(self, node: PsAstNode) -> set[PsTypedVariable]:
        """Returns all `PsTypedVariable`s that occur in the given AST without being defined prior to their usage."""

        undefined_vars: set[PsTypedVariable] = set()

        match node:
            case PsKernelFunction(block):
                return self(block)

            case PsExpression(expr):
                variables: set[Variable] = self._pb_dep_mapper(expr)

                for var in variables:
                    if not isinstance(var, PsTypedVariable):
                        raise PsMalformedAstException(
                            f"Non-typed variable {var} encountered"
                        )

                return cast(set[PsTypedVariable], variables)

            case PsAssignment(lhs, rhs):
                undefined_vars = self(lhs) | self(rhs)
                if isinstance(lhs.expression, PsTypedVariable):
                    undefined_vars.remove(lhs.expression)
                return undefined_vars

            case PsBlock(statements):
                for stmt in statements[::-1]:
                    undefined_vars -= self.declared_variables(stmt)
                    undefined_vars |= self(stmt)

                return undefined_vars

            case PsLoop(ctr, start, stop, step, body):
                undefined_vars = self(start) | self(stop) | self(step) | self(body)
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


def collect_undefined_variables(node: PsAstNode) -> set[PsTypedVariable]:
    return UndefinedVariablesCollector()(node)


class RequiredHeadersCollector(Collector):
    """Collect all header files required by a given AST for correct compilation.

    Required headers can currently only be defined in subclasses of `PsAbstractType`.
    """

    def __call__(self, node: PsAstNode) -> set[str]:
        match node:
            case PsExpression(expr):
                return self.rec(expr)
            case node:
                return reduce(set.union, (self(c) for c in node.children), set())

    def map_typed_variable(self, var: PsTypedVariable) -> set[str]:
        return var.dtype.required_headers

    def map_constant(self, cst: Any):
        if not isinstance(cst, PsTypedConstant):
            raise PsMalformedAstException("Untyped constant encountered in expression.")

        return cst.dtype.required_headers


def collect_required_headers(node: PsAstNode) -> set[str]:
    return RequiredHeadersCollector()(node)
