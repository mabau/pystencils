from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, overload, Callable, Any, cast
import operator

from ..symbols import PsSymbol
from ..constants import PsConstant
from ..literals import PsLiteral
from ..arrays import PsLinearizedArray, PsArrayBasePointer
from ..functions import PsFunction
from ...types import (
    PsType,
    PsScalarType,
    PsVectorType,
    PsTypeError,
)
from .util import failing_cast
from ..exceptions import PsInternalCompilerError

from .astnode import PsAstNode, PsLeafMixIn


class PsExpression(PsAstNode, ABC):
    """Base class for all expressions.

    **Types:** Each expression should be annotated with its type.
    Upon construction, the `dtype` property of most expression nodes is unset;
    only constant expressions, symbol expressions, and array accesses immediately inherit their type from
    their constant, symbol, or array, respectively.

    The canonical way to add types to newly constructed expressions is through the `Typifier`.
    It should be run at least once on any expression constructed by the backend.

    The type annotations are used by various transformation passes to make decisions, e.g. in
    function materialization and intrinsic selection.
    """

    def __init__(self, dtype: PsType | None = None) -> None:
        self._dtype = dtype

    @property
    def dtype(self) -> PsType | None:
        return self._dtype

    @dtype.setter
    def dtype(self, dt: PsType):
        self._dtype = dt

    def get_dtype(self) -> PsType:
        if self._dtype is None:
            raise PsInternalCompilerError("No dtype set on this expression yet.")

        return self._dtype

    def __add__(self, other: PsExpression) -> PsAdd:
        return PsAdd(self, other)

    def __sub__(self, other: PsExpression) -> PsSub:
        return PsSub(self, other)

    def __mul__(self, other: PsExpression) -> PsMul:
        return PsMul(self, other)

    def __truediv__(self, other: PsExpression) -> PsDiv:
        return PsDiv(self, other)

    def __neg__(self) -> PsNeg:
        return PsNeg(self)

    @overload
    @staticmethod
    def make(obj: PsSymbol) -> PsSymbolExpr:
        pass

    @overload
    @staticmethod
    def make(obj: PsConstant) -> PsConstantExpr:
        pass

    @overload
    @staticmethod
    def make(obj: PsLiteral) -> PsLiteralExpr:
        pass

    @staticmethod
    def make(obj: PsSymbol | PsConstant | PsLiteral) -> PsExpression:
        if isinstance(obj, PsSymbol):
            return PsSymbolExpr(obj)
        elif isinstance(obj, PsConstant):
            return PsConstantExpr(obj)
        elif isinstance(obj, PsLiteral):
            return PsLiteralExpr(obj)
        else:
            raise ValueError(f"Cannot make expression out of {obj}")

    @abstractmethod
    def clone(self) -> PsExpression:
        pass


class PsLvalue(ABC):
    """Mix-in for all expressions that may occur as an lvalue"""


class PsSymbolExpr(PsLeafMixIn, PsLvalue, PsExpression):
    """A single symbol as an expression."""

    __match_args__ = ("symbol",)

    def __init__(self, symbol: PsSymbol):
        super().__init__(symbol.dtype)
        self._symbol = symbol

    @property
    def symbol(self) -> PsSymbol:
        return self._symbol

    @symbol.setter
    def symbol(self, symbol: PsSymbol):
        self._symbol = symbol

    def clone(self) -> PsSymbolExpr:
        return PsSymbolExpr(self._symbol)

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsSymbolExpr):
            return False

        return self._symbol == other._symbol

    def __repr__(self) -> str:
        return f"Symbol({repr(self._symbol)})"


class PsConstantExpr(PsLeafMixIn, PsExpression):
    __match_args__ = ("constant",)

    def __init__(self, constant: PsConstant):
        super().__init__(constant.dtype)
        self._constant = constant

    @property
    def constant(self) -> PsConstant:
        return self._constant

    @constant.setter
    def constant(self, c: PsConstant):
        self._constant = c

    def clone(self) -> PsConstantExpr:
        return PsConstantExpr(self._constant)

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsConstantExpr):
            return False

        return self._constant == other._constant

    def __repr__(self) -> str:
        return f"PsConstantExpr({repr(self._constant)})"


class PsLiteralExpr(PsLeafMixIn, PsExpression):
    __match_args__ = ("literal",)

    def __init__(self, literal: PsLiteral):
        super().__init__(literal.dtype)
        self._literal = literal

    @property
    def literal(self) -> PsLiteral:
        return self._literal

    @literal.setter
    def literal(self, lit: PsLiteral):
        self._literal = lit

    def clone(self) -> PsLiteralExpr:
        return PsLiteralExpr(self._literal)

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsLiteralExpr):
            return False

        return self._literal == other._literal

    def __repr__(self) -> str:
        return f"PsLiteralExpr({repr(self._literal)})"


class PsSubscript(PsLvalue, PsExpression):
    __match_args__ = ("base", "index")

    def __init__(self, base: PsExpression, index: PsExpression):
        super().__init__()
        self._base = base
        self._index = index

    @property
    def base(self) -> PsExpression:
        return self._base

    @base.setter
    def base(self, expr: PsExpression):
        self._base = expr

    @property
    def index(self) -> PsExpression:
        return self._index

    @index.setter
    def index(self, expr: PsExpression):
        self._index = expr

    def clone(self) -> PsSubscript:
        return PsSubscript(self._base.clone(), self._index.clone())

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._base, self._index)

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0, 1][idx]
        match idx:
            case 0:
                self.base = failing_cast(PsExpression, c)
            case 1:
                self.index = failing_cast(PsExpression, c)

    def __repr__(self) -> str:
        return f"Subscript({self._base})[{self._index}]"


class PsArrayAccess(PsSubscript):
    __match_args__ = ("base_ptr", "index")

    def __init__(self, base_ptr: PsArrayBasePointer, index: PsExpression):
        super().__init__(PsExpression.make(base_ptr), index)
        self._base_ptr = base_ptr
        self._dtype = base_ptr.array.element_type

    @property
    def base_ptr(self) -> PsArrayBasePointer:
        return self._base_ptr

    @property
    def base(self) -> PsExpression:
        return self._base

    @base.setter
    def base(self, expr: PsExpression):
        if not isinstance(expr, PsSymbolExpr) or not isinstance(
            expr.symbol, PsArrayBasePointer
        ):
            raise ValueError(
                "Base expression of PsArrayAccess must be an array base pointer"
            )

        self._base_ptr = expr.symbol
        self._base = expr

    @property
    def array(self) -> PsLinearizedArray:
        return self._base_ptr.array

    def clone(self) -> PsArrayAccess:
        return PsArrayAccess(self._base_ptr, self._index.clone())

    def __repr__(self) -> str:
        return f"ArrayAccess({repr(self._base_ptr)}, {repr(self._index)})"


class PsVectorArrayAccess(PsArrayAccess):
    __match_args__ = ("base_ptr", "base_index")

    def __init__(
        self,
        base_ptr: PsArrayBasePointer,
        base_index: PsExpression,
        vector_entries: int,
        stride: int = 1,
        alignment: int = 0,
    ):
        super().__init__(base_ptr, base_index)
        element_type = base_ptr.array.element_type

        if not isinstance(element_type, PsScalarType):
            raise PsTypeError(
                "Cannot generate vector accesses to arrays with non-scalar elements"
            )

        self._vector_type = PsVectorType(
            element_type, vector_entries, const=element_type.const
        )
        self._stride = stride
        self._alignment = alignment

        self._dtype = self._vector_type

    @property
    def vector_entries(self) -> int:
        return self._vector_type.vector_entries

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def alignment(self) -> int:
        return self._alignment

    def get_vector_type(self) -> PsVectorType:
        return cast(PsVectorType, self._dtype)

    def clone(self) -> PsVectorArrayAccess:
        return PsVectorArrayAccess(
            self._base_ptr,
            self._index.clone(),
            self.vector_entries,
            self._stride,
            self._alignment,
        )

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsVectorArrayAccess):
            return False

        return (
            super().structurally_equal(other)
            and self._vector_type == other._vector_type
            and self._stride == other._stride
            and self._alignment == other._alignment
        )


class PsLookup(PsExpression, PsLvalue):
    __match_args__ = ("aggregate", "member_name")

    def __init__(self, aggregate: PsExpression, member_name: str) -> None:
        super().__init__()
        self._aggregate = aggregate
        self._member_name = member_name

    @property
    def aggregate(self) -> PsExpression:
        return self._aggregate

    @aggregate.setter
    def aggregate(self, aggr: PsExpression):
        self._aggregate = aggr

    @property
    def member_name(self) -> str:
        return self._member_name

    @member_name.setter
    def member_name(self, name: str):
        self._name = name

    def clone(self) -> PsLookup:
        return PsLookup(self._aggregate.clone(), self._member_name)

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._aggregate,)

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0][idx]
        self._aggregate = failing_cast(PsExpression, c)


class PsCall(PsExpression):
    __match_args__ = ("function", "args")

    def __init__(self, function: PsFunction, args: Sequence[PsExpression]) -> None:
        if len(args) != function.arg_count:
            raise ValueError(
                f"Argument count mismatch: Cannot apply function {function} to {len(args)} arguments."
            )

        super().__init__()

        self._function = function
        self._args = list(args)

    @property
    def function(self) -> PsFunction:
        return self._function

    @function.setter
    def function(self, func: PsFunction):
        if func.arg_count != self._function.arg_count:
            raise ValueError(
                "Current and replacement function must have the same number of parameters."
            )
        self._function = func

    @property
    def args(self) -> tuple[PsExpression, ...]:
        return tuple(self._args)

    @args.setter
    def args(self, exprs: Sequence[PsExpression]):
        if len(exprs) != self._function.arg_count:
            raise ValueError(
                f"Argument count mismatch: Cannot apply function {self._function} to {len(exprs)} arguments."
            )

        self._args = list(exprs)

    def clone(self) -> PsCall:
        return PsCall(self._function, [arg.clone() for arg in self._args])

    def get_children(self) -> tuple[PsAstNode, ...]:
        return self.args

    def set_child(self, idx: int, c: PsAstNode):
        self._args[idx] = failing_cast(PsExpression, c)

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsCall):
            return False
        return super().structurally_equal(other) and self._function == other._function

    def __str__(self):
        args = ", ".join(str(arg) for arg in self._args)
        return f"PsCall({self._function}, ({args}))"


class PsTernary(PsExpression):
    """Ternary operator."""

    __match_args__ = ("condition", "case_then", "case_else")

    def __init__(
        self, cond: PsExpression, then: PsExpression, els: PsExpression
    ) -> None:
        super().__init__()
        self._cond = cond
        self._then = then
        self._else = els

    @property
    def condition(self) -> PsExpression:
        return self._cond

    @property
    def case_then(self) -> PsExpression:
        return self._then

    @property
    def case_else(self) -> PsExpression:
        return self._else

    def clone(self) -> PsExpression:
        return PsTernary(self._cond.clone(), self._then.clone(), self._else.clone())

    def get_children(self) -> tuple[PsExpression, ...]:
        return (self._cond, self._then, self._else)

    def set_child(self, idx: int, c: PsAstNode):
        idx = range(3)[idx]
        match idx:
            case 0:
                self._cond = failing_cast(PsExpression, c)
            case 1:
                self._then = failing_cast(PsExpression, c)
            case 2:
                self._else = failing_cast(PsExpression, c)

    def __str__(self) -> str:
        return f"PsTernary({self._cond}, {self._then}, {self._else})"

    def __repr__(self) -> str:
        return f"PsTernary({repr(self._cond)}, {repr(self._then)}, {repr(self._else)})"


class PsNumericOpTrait:
    """Trait for operations valid only on numerical types"""


class PsIntOpTrait:
    """Trait for operations valid only on integer types"""


class PsBoolOpTrait:
    """Trait for boolean operations"""


class PsUnOp(PsExpression):
    __match_args__ = ("operand",)

    def __init__(self, operand: PsExpression):
        super().__init__()
        self._operand = operand

    @property
    def operand(self) -> PsExpression:
        return self._operand

    @operand.setter
    def operand(self, expr: PsExpression):
        self._operand = expr

    def clone(self) -> PsUnOp:
        return type(self)(self._operand.clone())

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._operand,)

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0][idx]
        self._operand = failing_cast(PsExpression, c)

    @property
    def python_operator(self) -> None | Callable[[Any], Any]:
        return None

    def __repr__(self) -> str:
        opname = self.__class__.__name__
        return f"{opname}({repr(self._operand)})"


class PsNeg(PsUnOp, PsNumericOpTrait):
    @property
    def python_operator(self):
        return operator.neg


class PsDeref(PsLvalue, PsUnOp):
    pass


class PsAddressOf(PsUnOp):
    pass


class PsCast(PsUnOp):
    __match_args__ = ("target_type", "operand")

    def __init__(self, target_type: PsType, operand: PsExpression):
        super().__init__(operand)
        self._target_type = target_type

    @property
    def target_type(self) -> PsType:
        return self._target_type

    @target_type.setter
    def target_type(self, dtype: PsType):
        self._target_type = dtype

    def clone(self) -> PsUnOp:
        return PsCast(self._target_type, self._operand.clone())

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsCast):
            return False
        return (
            super().structurally_equal(other)
            and self._target_type == other._target_type
        )


class PsBinOp(PsExpression):
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

    def clone(self) -> PsBinOp:
        return type(self)(self._op1.clone(), self._op2.clone())

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._op1, self._op2)

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


class PsAdd(PsBinOp, PsNumericOpTrait):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.add


class PsSub(PsBinOp, PsNumericOpTrait):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.sub


class PsMul(PsBinOp, PsNumericOpTrait):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.mul


class PsDiv(PsBinOp, PsNumericOpTrait):
    #  python_operator not implemented because can't unambigously decide
    #  between intdiv and truediv
    pass


class PsIntDiv(PsBinOp, PsIntOpTrait):
    """C-like integer division (round to zero)."""

    @property
    def python_operator(self) -> Callable[[Any, Any], Any]:
        from ...utils import c_intdiv

        return c_intdiv


class PsRem(PsBinOp, PsIntOpTrait):
    """C-style integer division remainder"""

    @property
    def python_operator(self) -> Callable[[Any, Any], Any]:
        from ...utils import c_rem

        return c_rem


class PsLeftShift(PsBinOp, PsIntOpTrait):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.lshift


class PsRightShift(PsBinOp, PsIntOpTrait):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.rshift


class PsBitwiseAnd(PsBinOp, PsIntOpTrait):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.and_


class PsBitwiseXor(PsBinOp, PsIntOpTrait):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.xor


class PsBitwiseOr(PsBinOp, PsIntOpTrait):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.or_


class PsAnd(PsBinOp, PsBoolOpTrait):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.and_


class PsOr(PsBinOp, PsBoolOpTrait):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.or_


class PsNot(PsUnOp, PsBoolOpTrait):
    @property
    def python_operator(self) -> Callable[[Any], Any] | None:
        return operator.not_


class PsRel(PsBinOp):
    """Base class for binary relational operators"""


class PsEq(PsRel):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.eq


class PsNe(PsRel):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.ne


class PsGe(PsRel):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.ge


class PsLe(PsRel):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.le


class PsGt(PsRel):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.gt


class PsLt(PsRel):
    @property
    def python_operator(self) -> Callable[[Any, Any], Any] | None:
        return operator.lt


class PsArrayInitList(PsExpression):
    __match_args__ = ("items",)

    def __init__(self, items: Sequence[PsExpression]):
        super().__init__()
        self._items = list(items)

    @property
    def items(self) -> list[PsExpression]:
        return self._items

    def get_children(self) -> tuple[PsAstNode, ...]:
        return tuple(self._items)

    def set_child(self, idx: int, c: PsAstNode):
        self._items[idx] = failing_cast(PsExpression, c)

    def clone(self) -> PsExpression:
        return PsArrayInitList([expr.clone() for expr in self._items])

    def __repr__(self) -> str:
        return f"PsArrayInitList({repr(self._items)})"


def evaluate_expression(
    expr: PsExpression, valuation: dict[str, Any]
) -> Any:
    """Evaluate a pystencils backend expression tree with values assigned to symbols according to the given valuation.

    Only a subset of expression nodes can be processed by this evaluator.
    """

    def visit(node):
        match node:
            case PsSymbolExpr(symb):
                return valuation[symb.name]

            case PsConstantExpr(c):
                return c.value

            case PsUnOp(op1) if node.python_operator is not None:
                return node.python_operator(visit(op1))

            case PsBinOp(op1, op2) if node.python_operator is not None:
                return node.python_operator(visit(op1), visit(op2))

            case other:
                raise NotImplementedError(
                    f"Unable to evaluate {other}: No implementation available."
                )

    return visit(expr)
