from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence, cast
from types import NoneType

from .astnode import PsAstNode, PsLeafMixIn
from .expressions import PsExpression, PsLvalue, PsSymbolExpr
from ..memory import PsSymbol

from .util import failing_cast


class PsStructuralNode(PsAstNode, ABC):
    """Base class for structural nodes in the pystencils AST.

    This class acts as a trait that structural AST nodes like blocks, conditionals, etc. can inherit from.
    """

    def clone(self):
        """Clone this structure node.

        .. note::
            Subclasses of `PsStructuralNode` should not override this method,
            but implement `_clone_structural` instead.
            That implementation shall call `clone` on any of its children.
        """
        return self._clone_structural()

    @abstractmethod
    def _clone_structural(self) -> PsStructuralNode:
        """Implementation of structural node cloning.

        :meta public:
        """
        pass


class PsBlock(PsStructuralNode):
    __match_args__ = ("statements",)

    def __init__(self, cs: Iterable[PsStructuralNode]):
        self._statements = list(cs)

    @property
    def children(self) -> Sequence[PsAstNode]:
        return self.get_children()

    @children.setter
    def children(self, cs: Sequence[PsAstNode]):
        self._statements = list([failing_cast(PsStructuralNode, c) for c in cs])

    def get_children(self) -> tuple[PsAstNode, ...]:
        return tuple(self._statements)

    def set_child(self, idx: int, c: PsAstNode):
        self._statements[idx] = failing_cast(PsStructuralNode, c)

    def _clone_structural(self) -> PsBlock:
        return PsBlock([stmt._clone_structural() for stmt in self._statements])

    @property
    def statements(self) -> list[PsStructuralNode]:
        return self._statements

    @statements.setter
    def statements(self, stm: Sequence[PsStructuralNode]):
        self._statements = list(stm)

    def __repr__(self) -> str:
        contents = ", ".join(repr(c) for c in self.children)
        return f"PsBlock( {contents} )"


class PsStatement(PsStructuralNode):
    __match_args__ = ("expression",)

    def __init__(self, expr: PsExpression):
        self._expression = expr

    @property
    def expression(self) -> PsExpression:
        return self._expression

    @expression.setter
    def expression(self, expr: PsExpression):
        self._expression = expr

    def _clone_structural(self) -> PsStatement:
        return PsStatement(self._expression.clone())

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._expression,)

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0][idx]
        assert idx == 0
        self._expression = failing_cast(PsExpression, c)


class PsAssignment(PsStructuralNode):
    __match_args__ = (
        "lhs",
        "rhs",
    )

    def __init__(self, lhs: PsExpression, rhs: PsExpression):
        if not isinstance(lhs, PsLvalue):
            raise ValueError("Assignment LHS must be an lvalue")
        self._lhs: PsExpression = lhs
        self._rhs = rhs

    @property
    def lhs(self) -> PsExpression:
        return self._lhs

    @lhs.setter
    def lhs(self, lvalue: PsExpression):
        if not isinstance(lvalue, PsLvalue):
            raise ValueError("Assignment LHS must be an lvalue")
        self._lhs = lvalue

    @property
    def rhs(self) -> PsExpression:
        return self._rhs

    @rhs.setter
    def rhs(self, expr: PsExpression):
        self._rhs = expr

    def _clone_structural(self) -> PsAssignment:
        return PsAssignment(self._lhs.clone(), self._rhs.clone())

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._lhs, self._rhs)

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0, 1][idx]  # trick to normalize index
        if idx == 0:
            self.lhs = failing_cast(PsExpression, c)
        elif idx == 1:
            self._rhs = failing_cast(PsExpression, c)
        else:
            assert False, "unreachable code"

    def __repr__(self) -> str:
        return f"PsAssignment({repr(self._lhs)}, {repr(self._rhs)})"


class PsDeclaration(PsAssignment):
    __match_args__ = (
        "lhs",
        "rhs",
    )

    def __init__(self, lhs: PsSymbolExpr, rhs: PsExpression):
        super().__init__(lhs, rhs)

    @property
    def lhs(self) -> PsExpression:
        return self._lhs

    @lhs.setter
    def lhs(self, lvalue: PsExpression):
        self._lhs = failing_cast(PsSymbolExpr, lvalue)

    @property
    def declared_symbol(self) -> PsSymbol:
        return cast(PsSymbolExpr, self._lhs).symbol

    def _clone_structural(self) -> PsDeclaration:
        return PsDeclaration(cast(PsSymbolExpr, self._lhs.clone()), self.rhs.clone())

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0, 1][idx]  # trick to normalize index
        if idx == 0:
            self.lhs = failing_cast(PsSymbolExpr, c)
        elif idx == 1:
            self._rhs = failing_cast(PsExpression, c)
        else:
            assert False, "unreachable code"

    def __repr__(self) -> str:
        return f"PsDeclaration({repr(self._lhs)}, {repr(self._rhs)})"


class PsLoop(PsStructuralNode):
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

    def _clone_structural(self) -> PsLoop:
        return PsLoop(
            self._ctr.clone(),
            self._start.clone(),
            self._stop.clone(),
            self._step.clone(),
            self._body._clone_structural(),
        )

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


class PsConditional(PsStructuralNode):
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

    def _clone_structural(self) -> PsConditional:
        return PsConditional(
            self._condition.clone(),
            self._branch_true._clone_structural(),
            self._branch_false._clone_structural() if self._branch_false is not None else None,
        )

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

    def __repr__(self) -> str:
        return f"PsConditional({repr(self._condition)}, {repr(self._branch_true)}, {repr(self._branch_false)})"


class PsEmptyLeafMixIn:
    """Mix-in marking AST leaves that can be treated as empty by the code generator,
    such as comments and preprocessor directives."""

    pass


class PsPragma(PsLeafMixIn, PsEmptyLeafMixIn, PsStructuralNode):
    """A C/C++ preprocessor pragma.

    Example usage: ``PsPragma("omp parallel for")`` translates to ``#pragma omp parallel for``.

    Args:
        text: The pragma's text, without the ``#pragma``.
    """

    __match_args__ = ("text",)

    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    def _clone_structural(self) -> PsPragma:
        return PsPragma(self.text)

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsPragma):
            return False

        return self._text == other._text


class PsComment(PsLeafMixIn, PsEmptyLeafMixIn, PsStructuralNode):
    __match_args__ = ("lines",)

    def __init__(self, text: str) -> None:
        self._text = text
        self._lines = tuple(text.splitlines())

    @property
    def text(self) -> str:
        return self._text

    @property
    def lines(self) -> tuple[str, ...]:
        return self._lines

    def _clone_structural(self) -> PsComment:
        return PsComment(self._text)

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsComment):
            return False

        return self._text == other._text
