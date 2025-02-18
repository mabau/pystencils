from ..kernelcreation import KernelCreationContext
from ..memory import PsSymbol
from ..exceptions import PsInternalCompilerError

from ..ast import PsAstNode
from ..ast.structural import (
    PsDeclaration,
    PsAssignment,
    PsLoop,
    PsConditional,
    PsBlock,
    PsStatement,
    PsEmptyLeafMixIn,
)
from ..ast.expressions import PsSymbolExpr, PsExpression

from ...types import constify

__all__ = ["CanonicalizeSymbols"]


class CanonContext:
    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx
        self.encountered_symbols: set[PsSymbol] = set()
        self.live_symbols_map: dict[PsSymbol, PsSymbol] = dict()

        self.updated_symbols: set[PsSymbol] = set()

    def deduplicate(self, symb: PsSymbol) -> PsSymbol:
        if symb in self.live_symbols_map:
            return self.live_symbols_map[symb]
        elif symb not in self.encountered_symbols:
            self.encountered_symbols.add(symb)
            self.live_symbols_map[symb] = symb
            return symb
        else:
            replacement = self._ctx.duplicate_symbol(symb)
            self.live_symbols_map[symb] = replacement
            self.encountered_symbols.add(replacement)
            return replacement

    def mark_as_updated(self, symb: PsSymbol):
        self.updated_symbols.add(symb)

    def is_live(self, symb: PsSymbol) -> bool:
        return symb in self.live_symbols_map

    def end_lifespan(self, symb: PsSymbol):
        if symb in self.live_symbols_map:
            del self.live_symbols_map[symb]


class CanonicalizeSymbols:
    """Remove duplicate symbol declarations and declare all non-updated symbols ``const``.

    The `CanonicalizeSymbols` pass will remove multiple declarations of the same symbol by
    renaming all but the last occurence, and will optionally ``const``-qualify all symbols
    encountered in the AST that are never updated.
    """

    def __init__(self, ctx: KernelCreationContext, constify: bool = True) -> None:
        self._ctx = ctx
        self._constify = constify
        self._last_result: CanonContext | None = None

    def get_last_live_symbols(self) -> set[PsSymbol]:
        if self._last_result is None:
            raise PsInternalCompilerError("Pass was not executed yet")
        return set(self._last_result.live_symbols_map.values())

    def __call__(self, node: PsAstNode) -> PsAstNode:
        cc = CanonContext(self._ctx)
        self.visit(node, cc)

        #   Any symbol encountered but never updated can be marked const
        if self._constify:
            for symb in cc.encountered_symbols - cc.updated_symbols:
                if symb.dtype is not None:
                    symb.dtype = constify(symb.dtype)

        #   Any symbols still alive now are function params or globals
        self._last_result = cc

        return node

    def visit(self, node: PsAstNode, cc: CanonContext):
        """Traverse the AST in reverse pre-order to collect, deduplicate, and maybe constify all live symbols."""

        match node:
            case PsSymbolExpr(symb):
                node.symbol = cc.deduplicate(symb)
                return node

            case PsExpression():
                for c in node.children:
                    self.visit(c, cc)

            case PsDeclaration(lhs, rhs):
                decl_symb = node.declared_symbol
                self.visit(lhs, cc)
                self.visit(rhs, cc)
                cc.end_lifespan(decl_symb)

            case PsAssignment(lhs, rhs):
                self.visit(lhs, cc)
                self.visit(rhs, cc)

                if isinstance(lhs, PsSymbolExpr):
                    cc.mark_as_updated(lhs.symbol)

            case PsLoop(ctr, _, _, _, _):
                decl_symb = ctr.symbol
                for c in node.children[::-1]:
                    self.visit(c, cc)
                cc.mark_as_updated(ctr.symbol)
                cc.end_lifespan(decl_symb)

            case PsConditional(cond, then, els):
                if els is not None:
                    self.visit(els, cc)
                self.visit(then, cc)
                self.visit(cond, cc)

            case PsBlock(statements):
                for stmt in statements[::-1]:
                    self.visit(stmt, cc)

            case PsStatement(expr):
                self.visit(expr, cc)

            case PsEmptyLeafMixIn():
                ...

            case unknown:
                raise PsInternalCompilerError(
                    f"Can't canonicalize symbols in {unknown} ({repr(unknown)})."
                )
