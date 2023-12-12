from __future__ import annotations
from typing import Sequence, Generator

from abc import ABC

import pymbolic.primitives as pb

from ..typed_expressions import PsTypedSymbol, PsLvalue


class PsAstNode(ABC):
    """Base class for all nodes in the pystencils AST."""

    def __init__(self, *children: Sequence[PsAstNode]):
        for c in children:
            if not isinstance(c, PsAstNode):
                raise TypeError(f"Child {c} was not a PsAstNode.")
        self._children = list(children)

    @property
    def children(self) -> Generator[PsAstNode, None, None]:
        yield from self._children

    def child(self, idx: int):
        return self._children[idx]

    @children.setter
    def children(self, cs: Sequence[PsAstNode]):
        if len(cs) != len(self._children):
            raise ValueError("The number of child nodes must remain the same!")
        self._children = list(cs)

    def __getitem__(self, idx: int):
        return self._children[idx]

    def __setitem__(self, idx: int, c: PsAstNode):
        self._children[idx] = c


class PsBlock(PsAstNode):

    @property
    def children(self) -> Generator[PsAstNode, None, None]:
        yield from self._children  # need to override entire property to override the setter

    @children.setter
    def children(self, cs: Sequence[PsAstNode]):
        self._children = cs


class PsExpression(PsAstNode):
    """Wrapper around pymbolics expressions."""

    def __init__(self, expr: pb.Expression):
        super().__init__()
        self._expr = expr

    @property
    def expression(self) -> pb.Expression:
        return self._expr

    @expression.setter
    def expression(self, expr: pb.Expression):
        self._expr = expr


class PsLvalueExpr(PsExpression):
    """Wrapper around pymbolics expressions that may occur at the left-hand side of an assignment"""

    def __init__(self, expr: PsLvalue):
        if not isinstance(expr, PsLvalue):
            raise TypeError("Expression was not a valid lvalue")

        super(PsLvalueExpr, self).__init__(expr)


class PsSymbolExpr(PsLvalueExpr):
    """Wrapper around PsTypedSymbols"""

    def __init__(self, symbol: PsTypedSymbol):
        if not isinstance(symbol, PsTypedSymbol):
            raise TypeError("Not a symbol!")

        super(PsLvalueExpr, self).__init__(symbol)

    @property
    def symbol(self) -> PsSymbolExpr:
        return self.expression

    @symbol.setter
    def symbol(self, symbol: PsSymbolExpr):
        self.expression = symbol


class PsAssignment(PsAstNode):
    def __init__(self, lhs: PsLvalueExpr, rhs: PsExpression):
        super(PsAssignment, self).__init__(lhs, rhs)

    @property
    def lhs(self) -> PsLvalueExpr:
        return self._children[0]

    @lhs.setter
    def lhs(self, lvalue: PsLvalueExpr):
        self._children[0] = lvalue

    @property
    def rhs(self) -> PsExpression:
        return self._children[1]

    @rhs.setter
    def rhs(self, expr: PsExpression):
        self._children[1] = expr


class PsDeclaration(PsAssignment):
    def __init__(self, lhs: PsSymbolExpr, rhs: PsExpression):
        super(PsDeclaration, self).__init__(lhs, rhs)

    @property
    def lhs(self) -> PsSymbolExpr:
        return self._children[0]

    @lhs.setter
    def lhs(self, symbol_node: PsSymbolExpr):
        self._children[0] = symbol_node


class PsLoop(PsAstNode):
    def __init__(self,
                 ctr: PsSymbolExpr,
                 start: PsExpression,
                 stop: PsExpression,
                 step: PsExpression,
                 body: PsBlock):
        super(PsLoop, self).__init__(ctr, start, stop, step, body)

    @property
    def counter(self) -> PsSymbolExpr:
        return self._children[0]

    @property
    def start(self) -> PsExpression:
        return self._children[1]

    @start.setter
    def start(self, expr: PsExpression):
        self._children[1] = expr

    @property
    def stop(self) -> PsExpression:
        return self._children[2]

    @stop.setter
    def stop(self, expr: PsExpression):
        self._children[2] = expr

    @property
    def step(self) -> PsExpression:
        return self._children[3]

    @step.setter
    def step(self, expr: PsExpression):
        self._children[3] = expr

    @property
    def body(self) -> PsBlock:
        return self._children[4]

    @body.setter
    def body(self, block: PsBlock):
        self._children[4] = block
