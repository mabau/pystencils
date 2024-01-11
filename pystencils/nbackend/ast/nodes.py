from __future__ import annotations
from typing import Sequence, Generator, TypeVar, Iterable, cast

from abc import ABC, abstractmethod

import pymbolic.primitives as pb

from ..typed_expressions import PsTypedVariable, PsArrayAccess, PsLvalue


T = TypeVar("T")


def failing_cast(target: type, obj: T):
    if not isinstance(obj, target):
        raise TypeError(f"Casting {obj} to {target} failed.")
    return obj


class PsAstNode(ABC):
    """Base class for all nodes in the pystencils AST.

    This base class provides a common interface to inspect and update the AST's branching structure.
    The four methods `num_children`, `children`, `get_child` and `set_child` must be implemented by
    each subclass.
    Subclasses are also responsible for doing the necessary type checks if they place restrictions on
    the types of their children.
    """

    @abstractmethod
    def num_children(self) -> int:
        ...

    @abstractmethod
    def children(self) -> Generator[PsAstNode, None, None]:
        ...

    @abstractmethod
    def get_child(self, idx: int):
        ...

    @abstractmethod
    def set_child(self, idx: int, c: PsAstNode):
        ...

    def set_children(self, cs: Iterable[PsAstNode]):
        for i, c in enumerate(cs):
            self.set_child(i, c)


class PsBlock(PsAstNode):
    def __init__(self, cs: Sequence[PsAstNode]):
        self._children = list(cs)

    def num_children(self) -> int:
        return len(self._children)

    def children(self) -> Generator[PsAstNode, None, None]:
        yield from self._children

    def get_child(self, idx: int):
        return self._children[idx]

    def set_child(self, idx: int, c: PsAstNode):
        self._children[idx] = c


class PsLeafNode(PsAstNode):
    def num_children(self) -> int:
        return 0

    def children(self) -> Generator[PsAstNode, None, None]:
        yield from ()

    def get_child(self, idx: int):
        raise IndexError("Child index out of bounds: Leaf nodes have no children.")

    def set_child(self, idx: int, c: PsAstNode):
        raise IndexError("Child index out of bounds: Leaf nodes have no children.")


class PsExpression(PsLeafNode):
    """Wrapper around pymbolics expressions."""

    def __init__(self, expr: pb.Expression):
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
        if not isinstance(expr, (PsTypedVariable, PsArrayAccess)):
            raise TypeError("Expression was not a valid lvalue")

        super(PsLvalueExpr, self).__init__(expr)


class PsSymbolExpr(PsLvalueExpr):
    """Wrapper around PsTypedSymbols"""

    def __init__(self, symbol: PsTypedVariable):
        super().__init__(symbol)

    @property
    def symbol(self) -> PsTypedVariable:
        return cast(PsTypedVariable, self._expr)

    @symbol.setter
    def symbol(self, symbol: PsTypedVariable):
        self._expr = symbol


class PsAssignment(PsAstNode):
    def __init__(self, lhs: PsLvalueExpr, rhs: PsExpression):
        self._lhs = lhs
        self._rhs = rhs

    @property
    def lhs(self) -> PsLvalueExpr:
        return self._lhs

    @lhs.setter
    def lhs(self, lvalue: PsLvalueExpr):
        self._lhs = lvalue

    @property
    def rhs(self) -> PsExpression:
        return self._rhs

    @rhs.setter
    def rhs(self, expr: PsExpression):
        self._rhs = expr

    def num_children(self) -> int:
        return 2

    def children(self) -> Generator[PsAstNode, None, None]:
        yield from (self._lhs, self._rhs)

    def get_child(self, idx: int):
        return (self._lhs, self._rhs)[idx]

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0, 1][idx]  # trick to normalize index
        if idx == 0:
            self._lhs = failing_cast(PsLvalueExpr, c)
        elif idx == 1:
            self._rhs = failing_cast(PsExpression, c)
        else:
            assert False, "unreachable code"


class PsDeclaration(PsAssignment):
    def __init__(self, lhs: PsSymbolExpr, rhs: PsExpression):
        super().__init__(lhs, rhs)

    @property
    def lhs(self) -> PsLvalueExpr:
        return self._lhs

    @lhs.setter
    def lhs(self, lvalue: PsLvalueExpr):
        self._lhs = failing_cast(PsSymbolExpr, lvalue)

    @property
    def declared_symbol(self) -> PsSymbolExpr:
        return cast(PsSymbolExpr, self._lhs)

    @declared_symbol.setter
    def declared_symbol(self, lvalue: PsSymbolExpr):
        self._lhs = lvalue

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0, 1][idx]  # trick to normalize index
        if idx == 0:
            self._lhs = failing_cast(PsSymbolExpr, c)
        elif idx == 1:
            self._rhs = failing_cast(PsExpression, c)
        else:
            assert False, "unreachable code"


class PsLoop(PsAstNode):
    def __init__(
        self,
        ctr: PsSymbolExpr,
        start: PsExpression,
        stop: PsExpression,
        step: PsExpression,
        body: PsBlock,
    ):
        self._ctr = ctr
        self._start = start
        self._stop = stop
        self._step = step
        self._body = body

    @property
    def counter(self) -> PsSymbolExpr:
        return self._ctr

    @counter.setter
    def counter(self, expr: PsSymbolExpr):
        self._ctr = expr

    @property
    def start(self) -> PsExpression:
        return self._start

    @start.setter
    def start(self, expr: PsExpression):
        self._start = expr

    @property
    def stop(self) -> PsExpression:
        return self._stop

    @stop.setter
    def stop(self, expr: PsExpression):
        self._stop = expr

    @property
    def step(self) -> PsExpression:
        return self._step

    @step.setter
    def step(self, expr: PsExpression):
        self._step = expr

    @property
    def body(self) -> PsBlock:
        return self._body

    @body.setter
    def body(self, block: PsBlock):
        self._body = block

    def num_children(self) -> int:
        return 5

    def children(self) -> Generator[PsAstNode, None, None]:
        yield from (self._ctr, self._start, self._stop, self._step, self._body)

    def get_child(self, idx: int):
        return (self._ctr, self._start, self._stop, self._step, self._body)[idx]

    def set_child(self, idx: int, c: PsAstNode):
        idx = list(range(5))[idx]
        match idx:
            case 0:
                self._ctr = failing_cast(PsSymbolExpr, c)
            case 1:
                self._start = failing_cast(PsExpression, c)
            case 2:
                self._stop = failing_cast(PsExpression, c)
            case 3:
                self._step = failing_cast(PsExpression, c)
            case 4:
                self._body = failing_cast(PsBlock, c)
            case _:
                assert False, "unreachable code"
