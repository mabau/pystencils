from typing import TypeVar, cast

from ..kernelcreation import KernelCreationContext
from ..memory import PsSymbol
from ..exceptions import PsInternalCompilerError

from ..ast import PsAstNode
from ..ast.structural import (
    PsBlock,
    PsConditional,
    PsLoop,
    PsDeclaration,
    PsAssignment,
    PsComment,
    PsPragma,
    PsStatement,
)
from ..ast.expressions import PsExpression, PsSymbolExpr

__all__ = ["CanonicalClone"]


class CloneContext:
    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx
        self._dup_table: dict[PsSymbol, PsSymbol] = dict()

    def symbol_decl(self, declared_symbol: PsSymbol):
        self._dup_table[declared_symbol] = self._ctx.duplicate_symbol(declared_symbol)

    def get_replacement(self, symb: PsSymbol):
        return self._dup_table.get(symb, symb)


Node_T = TypeVar("Node_T", bound=PsAstNode)


class CanonicalClone:
    """Clone a subtree, and rename all symbols declared inside it to retain canonicality."""

    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx

    def __call__(self, node: Node_T) -> Node_T:
        return self.visit(node, CloneContext(self._ctx))

    def visit(self, node: Node_T, cc: CloneContext) -> Node_T:
        match node:
            case PsBlock(statements):
                return cast(
                    Node_T, PsBlock([self.visit(stmt, cc) for stmt in statements])
                )

            case PsLoop(ctr, start, stop, step, body):
                cc.symbol_decl(ctr.symbol)
                return cast(
                    Node_T,
                    PsLoop(
                        self.visit(ctr, cc),
                        self.visit(start, cc),
                        self.visit(stop, cc),
                        self.visit(step, cc),
                        self.visit(body, cc),
                    ),
                )

            case PsConditional(cond, then, els):
                return cast(
                    Node_T,
                    PsConditional(
                        self.visit(cond, cc),
                        self.visit(then, cc),
                        self.visit(els, cc) if els is not None else None,
                    ),
                )

            case PsComment() | PsPragma():
                return cast(Node_T, node.clone())

            case PsDeclaration(lhs, rhs):
                cc.symbol_decl(node.declared_symbol)
                return cast(
                    Node_T,
                    PsDeclaration(
                        cast(PsSymbolExpr, self.visit(lhs, cc)),
                        self.visit(rhs, cc),
                    ),
                )

            case PsAssignment(lhs, rhs):
                return cast(
                    Node_T,
                    PsAssignment(
                        self.visit(lhs, cc),
                        self.visit(rhs, cc),
                    ),
                )

            case PsExpression():
                expr_clone = node.clone()
                self._replace_symbols(expr_clone, cc)
                return cast(Node_T, expr_clone)

            case PsStatement(expr):
                return cast(Node_T, PsStatement(self.visit(expr, cc)))

            case _:
                raise PsInternalCompilerError(
                    f"Don't know how to canonically clone {type(node)}"
                )

    def _replace_symbols(self, expr: PsExpression, cc: CloneContext):
        if isinstance(expr, PsSymbolExpr):
            expr.symbol = cc.get_replacement(expr.symbol)
        else:
            for c in expr.children:
                self._replace_symbols(cast(PsExpression, c), cc)
