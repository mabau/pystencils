from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, overload, Callable, Any, cast
import operator

import numpy as np
from numpy.typing import NDArray

from ..memory import PsSymbol, PsBuffer, BufferBasePtr
from ..constants import PsConstant
from ..literals import PsLiteral
from ..functions import PsFunction
from ...types import (
    PsType,
    PsVectorType,
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

    .. attention::
        The ``structurally_equal`` check currently does not take expression data types into
        account. This may change in the future.
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

    def clone(self):
        """Clone this expression.
        
        .. note::
            Subclasses of `PsExpression` should not override this method,
            but implement `_clone_expr` instead.
            That implementation shall call `clone` on any of its subexpressions,
            but does not need to fix the `dtype` property.
            The `dtype` is correctly applied by `PsExpression.clone` internally.
        """
        cloned = self._clone_expr()
        cloned._dtype = self.dtype
        return cloned

    @abstractmethod
    def _clone_expr(self) -> PsExpression:
        """Implementation of expression cloning.
        
        :meta public:
        """
        pass


class PsLvalue(ABC):
    """Mix-in for all expressions that may occur as an lvalue;
    i.e. expressions that represent a memory location."""


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

    def _clone_expr(self) -> PsSymbolExpr:
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

    def _clone_expr(self) -> PsConstantExpr:
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

    def _clone_expr(self) -> PsLiteralExpr:
        return PsLiteralExpr(self._literal)

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsLiteralExpr):
            return False

        return self._literal == other._literal

    def __repr__(self) -> str:
        return f"PsLiteralExpr({repr(self._literal)})"


class PsBufferAcc(PsLvalue, PsExpression):
    """Access into a `PsBuffer`."""

    __match_args__ = ("base_pointer", "index")

    def __init__(self, base_ptr: PsSymbol, index: Sequence[PsExpression]):
        super().__init__()
        bptr_prop = cast(BufferBasePtr, base_ptr.get_properties(BufferBasePtr).pop())

        if len(index) != bptr_prop.buffer.dim:
            raise ValueError("Number of index expressions must equal buffer shape.")

        self._base_ptr = PsExpression.make(base_ptr)
        self._index = list(index)
        self._dtype = bptr_prop.buffer.element_type

    @property
    def base_pointer(self) -> PsSymbolExpr:
        return self._base_ptr

    @base_pointer.setter
    def base_pointer(self, expr: PsSymbolExpr):
        bptr_prop = cast(BufferBasePtr, expr.symbol.get_properties(BufferBasePtr).pop())
        if bptr_prop.buffer != self.buffer:
            raise ValueError(
                "Cannot replace a buffer access's base pointer with one belonging to a different buffer."
            )

        self._base_ptr = expr

    @property
    def buffer(self) -> PsBuffer:
        return cast(
            BufferBasePtr, self._base_ptr.symbol.get_properties(BufferBasePtr).pop()
        ).buffer

    @property
    def index(self) -> list[PsExpression]:
        return self._index

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._base_ptr,) + tuple(self._index)

    def set_child(self, idx: int, c: PsAstNode):
        idx = range(len(self._index) + 1)[idx]
        if idx == 0:
            self.base_pointer = failing_cast(PsSymbolExpr, c)
        else:
            self._index[idx - 1] = failing_cast(PsExpression, c)

    def _clone_expr(self) -> PsBufferAcc:
        return PsBufferAcc(self._base_ptr.symbol, [i.clone() for i in self._index])

    def __repr__(self) -> str:
        return f"PsBufferAcc({repr(self._base_ptr)}, {repr(self._index)})"


class PsSubscript(PsLvalue, PsExpression):
    """N-dimensional subscript into an array."""

    __match_args__ = ("array", "index")

    def __init__(self, arr: PsExpression, index: Sequence[PsExpression]):
        super().__init__()
        self._arr = arr

        if not index:
            raise ValueError("Subscript index cannot be empty.")

        self._index = list(index)

    @property
    def array(self) -> PsExpression:
        return self._arr

    @array.setter
    def array(self, expr: PsExpression):
        self._arr = expr

    @property
    def index(self) -> list[PsExpression]:
        return self._index

    @index.setter
    def index(self, idx: Sequence[PsExpression]):
        self._index = list(idx)

    def _clone_expr(self) -> PsSubscript:
        return PsSubscript(self._arr.clone(), [i.clone() for i in self._index])

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._arr,) + tuple(self._index)

    def set_child(self, idx: int, c: PsAstNode):
        idx = range(len(self._index) + 1)[idx]
        match idx:
            case 0:
                self.array = failing_cast(PsExpression, c)
            case _:
                self.index[idx - 1] = failing_cast(PsExpression, c)

    def __repr__(self) -> str:
        idx = ", ".join(repr(i) for i in self._index)
        return f"PsSubscript({repr(self._arr)}, {repr(idx)})"


class PsMemAcc(PsLvalue, PsExpression):
    """Pointer-based memory access with type-dependent offset."""

    __match_args__ = ("pointer", "offset")

    def __init__(self, ptr: PsExpression, offset: PsExpression):
        super().__init__()
        self._ptr = ptr
        self._offset = offset

    @property
    def pointer(self) -> PsExpression:
        return self._ptr

    @pointer.setter
    def pointer(self, expr: PsExpression):
        self._ptr = expr

    @property
    def offset(self) -> PsExpression:
        return self._offset

    @offset.setter
    def offset(self, expr: PsExpression):
        self._offset = expr

    def _clone_expr(self) -> PsMemAcc:
        return PsMemAcc(self._ptr.clone(), self._offset.clone())

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._ptr, self._offset)

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0, 1][idx]
        match idx:
            case 0:
                self.pointer = failing_cast(PsExpression, c)
            case 1:
                self.offset = failing_cast(PsExpression, c)

    def __repr__(self) -> str:
        return f"PsMemAcc({repr(self._ptr)}, {repr(self._offset)})"


class PsVectorMemAcc(PsMemAcc):
    """Pointer-based vectorized memory access."""

    __match_args__ = ("base_ptr", "base_index")

    def __init__(
        self,
        base_ptr: PsExpression,
        base_index: PsExpression,
        vector_entries: int,
        stride: int = 1,
        alignment: int = 0,
    ):
        super().__init__(base_ptr, base_index)

        self._vector_entries = vector_entries
        self._stride = stride
        self._alignment = alignment

    @property
    def vector_entries(self) -> int:
        return self._vector_entries

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def alignment(self) -> int:
        return self._alignment

    def get_vector_type(self) -> PsVectorType:
        return cast(PsVectorType, self._dtype)

    def _clone_expr(self) -> PsVectorMemAcc:
        return PsVectorMemAcc(
            self._ptr.clone(),
            self._offset.clone(),
            self.vector_entries,
            self._stride,
            self._alignment,
        )

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsVectorMemAcc):
            return False

        return (
            super().structurally_equal(other)
            and self._vector_entries == other._vector_entries
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

    def _clone_expr(self) -> PsLookup:
        return PsLookup(self._aggregate.clone(), self._member_name)

    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._aggregate,)

    def set_child(self, idx: int, c: PsAstNode):
        idx = [0][idx]
        self._aggregate = failing_cast(PsExpression, c)

    def __repr__(self) -> str:
        return f"PsLookup({repr(self._aggregate)}, {repr(self._member_name)})"


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

    def _clone_expr(self) -> PsCall:
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

    def _clone_expr(self) -> PsExpression:
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

    def _clone_expr(self) -> PsUnOp:
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


class PsAddressOf(PsUnOp):
    """Take the address of a memory location.

    .. DANGER::
        Taking the address of a memory location owned by a symbol or field array
        introduces an alias to that memory location.
        As pystencils assumes its symbols and fields to never be aliased, this can
        subtly change the semantics of a kernel.
        Use the address-of operator with utmost care.
    """

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

    def _clone_expr(self) -> PsUnOp:
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

    def _clone_expr(self) -> PsBinOp:
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
    """N-dimensional array initialization matrix."""

    __match_args__ = ("items",)

    def __init__(
        self,
        items: Sequence[PsExpression | Sequence[PsExpression | Sequence[PsExpression]]],
    ):
        super().__init__()
        self._items = np.array(items, dtype=np.object_)

    @property
    def items_grid(self) -> NDArray[np.object_]:
        return self._items

    @property
    def shape(self) -> tuple[int, ...]:
        return self._items.shape

    @property
    def items(self) -> tuple[PsExpression, ...]:
        return tuple(self._items.flat)  # type: ignore

    def get_children(self) -> tuple[PsAstNode, ...]:
        return tuple(self._items.flat)  # type: ignore

    def set_child(self, idx: int, c: PsAstNode):
        self._items.flat[idx] = failing_cast(PsExpression, c)

    def _clone_expr(self) -> PsExpression:
        return PsArrayInitList(
            np.array([expr.clone() for expr in self.children]).reshape(  # type: ignore
                self._items.shape
            )
        )

    def __repr__(self) -> str:
        return f"PsArrayInitList({repr(self._items)})"


def evaluate_expression(expr: PsExpression, valuation: dict[str, Any]) -> Any:
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
