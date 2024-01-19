from __future__ import annotations
from typing import Sequence, Iterable, cast, TypeAlias
from types import NoneType

from abc import ABC, abstractmethod

from ..typed_expressions import PsTypedVariable, ExprOrConstant
from ..arrays import PsArrayAccess
from .util import failing_cast


class PsAstNode(ABC):
    """Base class for all nodes in the pystencils AST.

    This base class provides a common interface to inspect and update the AST's branching structure.
    The two methods `get_children` and `set_child` must be implemented by each subclass.
    Subclasses are also responsible for doing the necessary type checks if they place restrictions on
    the types of their children.
    """

    @property
    def children(self) -> tuple[PsAstNode, ...]:
        return self.get_children()

    @children.setter
    def children(self, cs: Iterable[PsAstNode]):
        for i, c in enumerate(cs):
            self.set_child(i, c)

    @abstractmethod
    def get_children(self) -> tuple[PsAstNode, ...]:
        ...

    @abstractmethod
    def set_child(self, idx: int, c: PsAstNode):
        ...


class PsBlock(PsAstNode):
    __match_args__ = ("statements",)

    def __init__(self, cs: Sequence[PsAstNode]):
        self._statements = list(cs)

    def get_children(self) -> tuple[PsAstNode, ...]:
        return tuple(self._statements)

    def set_child(self, idx: int, c: PsAstNode):
        self._statements[idx] = c

    @property
    def statements(self) -> list[PsAstNode]:
        return self._statements

    @statements.setter
    def statements(self, stm: Sequence[PsAstNode]):
        self._statements = list(stm)


class PsLeafNode(PsAstNode):
    def get_children(self) -> tuple[PsAstNode, ...]:
        return ()

    def set_child(self, idx: int, c: PsAstNode):
        raise IndexError("Child index out of bounds: Leaf nodes have no children.")


class PsExpression(PsLeafNode):
    """Wrapper around pymbolics expressions."""

    __match_args__ = ("expression",)

    def __init__(self, expr: ExprOrConstant):
        self._expr = expr

    @property
    def expression(self) -> ExprOrConstant:
        return self._expr

    @expression.setter
    def expression(self, expr: ExprOrConstant):
        self._expr = expr


class PsLvalueExpr(PsExpression):
    """Wrapper around pymbolics expressions that may occur at the left-hand side of an assignment"""

    def __init__(self, expr: PsLvalue):
        if not isinstance(expr, (PsTypedVariable, PsArrayAccess)):
            raise TypeError("Expression was not a valid lvalue")

        super(PsLvalueExpr, self).__init__(expr)


class PsSymbolExpr(PsLvalueExpr):
    """Wrapper around PsTypedSymbols"""

    __match_args__ = ("symbol",)

    def __init__(self, symbol: PsTypedVariable):
        super().__init__(symbol)

    @property
    def symbol(self) -> PsTypedVariable:
        return cast(PsTypedVariable, self._expr)

    @symbol.setter
    def symbol(self, symbol: PsTypedVariable):
        self._expr = symbol


PsLvalue: TypeAlias = PsTypedVariable | PsArrayAccess
"""Types of expressions that may occur on the left-hand side of assignments."""


class PsAssignment(PsAstNode):
    __match_args__ = (
        "lhs",
        "rhs",
    )

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

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._lhs, self._rhs)

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0, 1][idx]  # trick to normalize index
        if idx == 0:
            self._lhs = failing_cast(PsLvalueExpr, c)
        elif idx == 1:
            self._rhs = failing_cast(PsExpression, c)
        else:
            assert False, "unreachable code"


class PsDeclaration(PsAssignment):
    __match_args__ = (
        "declared_variable",
        "rhs",
    )

    def __init__(self, lhs: PsSymbolExpr, rhs: PsExpression):
        super().__init__(lhs, rhs)

    @property
    def lhs(self) -> PsLvalueExpr:
        return self._lhs

    @lhs.setter
    def lhs(self, lvalue: PsLvalueExpr):
        self._lhs = failing_cast(PsSymbolExpr, lvalue)

    @property
    def declared_variable(self) -> PsSymbolExpr:
        return cast(PsSymbolExpr, self._lhs)

    @declared_variable.setter
    def declared_variable(self, lvalue: PsSymbolExpr):
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
    __match_args__ = ("counter", "start", "stop", "step", "body")

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

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._ctr, self._start, self._stop, self._step, self._body)

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


class PsConditional(PsAstNode):
    """Conditional branch"""

    __match_args__ = ("condition", "branch_true", "branch_false")

    def __init__(
        self,
        cond: PsExpression,
        branch_true: PsBlock,
        branch_false: PsBlock | None = None,
    ):
        self._condition = cond
        self._branch_true = branch_true
        self._branch_false = branch_false

    @property
    def condition(self) -> PsExpression:
        return self._condition

    @condition.setter
    def condition(self, expr: PsExpression):
        self._condition = expr

    @property
    def branch_true(self) -> PsBlock:
        return self._branch_true

    @branch_true.setter
    def branch_true(self, block: PsBlock):
        self._branch_true = block

    @property
    def branch_false(self) -> PsBlock | None:
        return self._branch_false

    @branch_false.setter
    def branch_false(self, block: PsBlock | None):
        self._branch_false = block

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._condition, self._branch_true) + (
            (self._branch_false,) if self._branch_false is not None else ()
        )

    def set_child(self, idx: int, c: PsAstNode):
        idx = list(range(3))[idx]
        match idx:
            case 0:
                self._condition = failing_cast(PsExpression, c)
            case 1:
                self._branch_true = failing_cast(PsBlock, c)
            case 2:
                self._branch_false = failing_cast((PsBlock, NoneType), c)
            case _:
                assert False, "unreachable code"
