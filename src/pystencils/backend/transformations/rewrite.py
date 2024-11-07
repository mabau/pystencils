from typing import overload

from ..memory import PsSymbol
from ..ast import PsAstNode
from ..ast.structural import PsBlock
from ..ast.expressions import PsExpression, PsSymbolExpr


@overload
def substitute_symbols(node: PsBlock, subs: dict[PsSymbol, PsExpression]) -> PsBlock:
    pass


@overload
def substitute_symbols(
    node: PsExpression, subs: dict[PsSymbol, PsExpression]
) -> PsExpression:
    pass


@overload
def substitute_symbols(
    node: PsAstNode, subs: dict[PsSymbol, PsExpression]
) -> PsAstNode:
    pass


def substitute_symbols(
    node: PsAstNode, subs: dict[PsSymbol, PsExpression]
) -> PsAstNode:
    """Substitute expressions for symbols throughout a subtree."""
    match node:
        case PsSymbolExpr(symb) if symb in subs:
            return subs[symb].clone()
        case _:
            node.children = [substitute_symbols(c, subs) for c in node.children]
            return node
