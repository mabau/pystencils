from typing import cast
from functools import reduce

from .structural import (
    PsAstNode,
    PsExpression,
    PsStatement,
    PsAssignment,
    PsDeclaration,
    PsLoop,
    PsBlock,
)
from .expressions import PsSymbolExpr, PsConstantExpr

from ..symbols import PsSymbol
from ..exceptions import PsInternalCompilerError


class UndefinedSymbolsCollector:
    """Collector for undefined variables.

    This class implements an AST visitor that collects all `PsTypedVariable`s that have been used
    in the AST without being defined prior to their usage.
    """

    def __call__(self, node: PsAstNode) -> set[PsSymbol]:
        """Returns all `PsTypedVariable`s that occur in the given AST without being defined prior to their usage."""
        return self.visit(node)

    def visit(self, node: PsAstNode) -> set[PsSymbol]:
        undefined_vars: set[PsSymbol] = set()

        match node:
            case PsExpression():
                return self.visit_expr(node)

            case PsStatement(expr):
                return self.visit_expr(expr)

            case PsAssignment(lhs, rhs):
                undefined_vars = self(lhs) | self(rhs)
                if isinstance(lhs, PsSymbolExpr):
                    undefined_vars.remove(lhs.symbol)
                return undefined_vars

            case PsBlock(statements):
                for stmt in statements[::-1]:
                    undefined_vars -= self.declared_variables(stmt)
                    undefined_vars |= self(stmt)

                return undefined_vars

            case PsLoop(ctr, start, stop, step, body):
                undefined_vars = self(start) | self(stop) | self(step) | self(body)
                undefined_vars.discard(ctr.symbol)
                return undefined_vars

            case unknown:
                raise PsInternalCompilerError(
                    f"Don't know how to collect undefined variables from {unknown}"
                )

    def visit_expr(self, expr: PsExpression) -> set[PsSymbol]:
        match expr:
            case PsSymbolExpr(symb):
                return {symb}
            case _:
                return reduce(
                    set.union,
                    (self.visit_expr(cast(PsExpression, c)) for c in expr.children),
                    set(),
                )

    def declared_variables(self, node: PsAstNode) -> set[PsSymbol]:
        """Returns the set of variables declared by the given node which are visible in the enclosing scope."""

        match node:
            case PsDeclaration(lhs, _):
                return {lhs.symbol}

            case PsStatement() | PsAssignment() | PsExpression() | PsLoop() | PsBlock():
                return set()

            case unknown:
                raise PsInternalCompilerError(
                    f"Don't know how to collect declared variables from {unknown}"
                )


def collect_undefined_symbols(node: PsAstNode) -> set[PsSymbol]:
    return UndefinedSymbolsCollector()(node)


def collect_required_headers(node: PsAstNode) -> set[str]:
    match node:
        case PsSymbolExpr(symb):
            return symb.get_dtype().required_headers
        case PsConstantExpr(cs):
            return cs.get_dtype().required_headers
        case _:
            return reduce(
                set.union, (collect_required_headers(c) for c in node.children), set()
            )
