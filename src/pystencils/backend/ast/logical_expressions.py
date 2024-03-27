from typing import Callable, Any
import operator

from .expressions import PsExpression
from .astnode import PsAstNode
from .util import failing_cast


class PsLogicalExpression(PsExpression):
    __match_args__ = ("operand1", "operand2")

    def __init__(self, op1: PsExpression, op2: PsExpression):
        super().__init__()
        self._op1 = op1
        self._op2 = op2

    @property
    def operand1(self) -> PsExpression:
        return self._op1

    @operand1.setter
    def operand1(self, expr: PsExpression):
        self._op1 = expr

    @property
    def operand2(self) -> PsExpression:
        return self._op2

    @operand2.setter
    def operand2(self, expr: PsExpression):
        self._op2 = expr

    def clone(self):
        return type(self)(self._op1.clone(), self._op2.clone())

    def get_children(self) -> tuple[PsAstNode, ...]:
        return self._op1, self._op2

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0, 1][idx]
        match idx:
            case 0:
                self._op1 = failing_cast(PsExpression, c)
            case 1:
                self._op2 = failing_cast(PsExpression, c)

    def __repr__(self) -> str:
        opname = self.__class__.__name__
        return f"{opname}({repr(self._op1)}, {repr(self._op2)})"

    @property
    def python_operator(self) -> None | Callable[[Any, Any], Any]:
        return None


class PsAnd(PsLogicalExpression):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.and_


class PsEq(PsLogicalExpression):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.eq


class PsGe(PsLogicalExpression):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.ge


class PsGt(PsLogicalExpression):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.gt


class PsLe(PsLogicalExpression):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.le


class PsLt(PsLogicalExpression):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.lt


class PsNe(PsLogicalExpression):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.ne
