from __future__ import annotations
from typing import Sequence, cast
from types import NoneType

from .astnode import PsAstNode, PsLeafMixIn
from .expressions import PsExpression, PsLvalueExpr, PsSymbolExpr

from .util import failing_cast


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

    def __repr__(self) -> str:
        contents = ", ".join(repr(c) for c in self.children)
        return f"PsBlock( {contents} )"


class PsStatement(PsAstNode):
    __match_args__ = ("expression",)

    def __init__(self, expr: PsExpression):
        self._expression = expr

    @property
    def expression(self) -> PsExpression:
        return self._expression

    @expression.setter
    def expression(self, expr: PsExpression):
        self._expression = expr

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._expression,)

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0][idx]
        assert idx == 0
        self._expression = failing_cast(PsExpression, c)


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

    def __repr__(self) -> str:
        return f"PsAssignment({repr(self._lhs)}, {repr(self._rhs)})"


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

    def __repr__(self) -> str:
        return f"PsDeclaration({repr(self._lhs)}, {repr(self._rhs)})"


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


class PsComment(PsLeafMixIn, PsAstNode):
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

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsComment):
            return False

        return self._text == other._text
