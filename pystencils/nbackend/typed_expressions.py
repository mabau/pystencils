from __future__ import annotations

from functools import reduce
from typing import TypeAlias, Union, Any, Tuple, Callable
import operator

import pymbolic.primitives as pb

from .types import (
    PsAbstractType,
    PsScalarType,
    PsNumericType,
    PsPointerType,
    constify,
    PsTypeError,
)


class PsTypedVariable(pb.Variable):
    def __init__(self, name: str, dtype: PsAbstractType):
        super(PsTypedVariable, self).__init__(name)
        self._dtype = dtype

    @property
    def dtype(self) -> PsAbstractType:
        return self._dtype


class PsArray:
    def __init__(
        self,
        name: str,
        length: pb.Expression,
        element_type: PsScalarType,  # todo Frederik: is PsScalarType correct?
    ):
        self._name = name
        self._length = length
        self._element_type = element_type

    @property
    def name(self):
        return self._name

    @property
    def length(self):
        return self._length

    @property
    def element_type(self):
        return self._element_type


class PsLinearizedArray(PsArray):
    """N-dimensional contiguous array"""

    def __init__(
        self,
        name: str,
        shape: Tuple[pb.Expression, ...],
        strides: Tuple[pb.Expression],
        element_type: PsScalarType,
    ):
        length = reduce(lambda x, y: x * y, shape, 1)
        super().__init__(name, length, element_type)

        self._shape = shape
        self._strides = strides

    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides


class PsArrayBasePointer(PsTypedVariable):
    def __init__(self, name: str, array: PsArray):
        dtype = PsPointerType(array.element_type)
        super().__init__(name, dtype)

        self._array = array

    @property
    def array(self):
        return self._array


class PsArrayAccess(pb.Subscript):
    def __init__(self, base_ptr: PsArrayBasePointer, index: pb.Expression):
        super(PsArrayAccess, self).__init__(base_ptr, index)
        self._base_ptr = base_ptr
        self._index = index

    @property
    def base_ptr(self):
        return self._base_ptr

    # @property
    # def index(self):
    #     return self._index

    @property
    def array(self) -> PsArray:
        return self._base_ptr.array

    @property
    def dtype(self) -> PsAbstractType:
        """Data type of this expression, i.e. the element type of the underlying array"""
        return self._base_ptr.array.element_type


PsLvalue: TypeAlias = Union[PsTypedVariable, PsArrayAccess]


class PsTypedConstant:
    """Represents typed constants occuring in the pystencils AST.

    A Note On Divisions
    -------------------

    The semantics of divisions in C and Python differ greatly.
    Python has two division operators: `/` (`truediv`) and `//` (`floordiv`).
    `truediv` is pure floating-point division, and so applied to floating-point numbers maps exactly to
    floating-point division in C, but not when applied to integers.
    `floordiv` has no C equivalent:
    While `floordiv` performs euclidean division and always rounds its result
    downward (`3 // 2 == 1`, and `-3 // 2 = -2`),
    the C `/` operator on integers always rounds toward zero (in C, `-3 / 2 = -1`.)

    The same applies to the `%` operator:
    In Python, `%` computes the euclidean modulus (e.g. `-3 % 2 = 1`),
    while in C, `%` computes the remainder (e.g. `-3 % 2 = -1`).
    These two differ whenever negative numbers are involved.

    Pymbolic provides `Quotient` to model Python's `/`, `FloorDiv` to model `//`, and `Remainder` to model `%`.
    The last one is a misnomer: it should instead be called `Modulus`.

    Since the pystencils backend has to accurately capture the behaviour of C,
    the behaviour of `/` is overridden in `PsTypedConstant`.
    In a floating-point context, it behaves as usual, while in an integer context,
    it implements C-style integer division.
    Similarily, `%` is only legal in integer contexts, where it implements the C-style remainder.
    Usage of `//` and the pymbolic `FloorDiv` is illegal.
    """

    @staticmethod
    def try_create(value: Any, dtype: PsNumericType):
        try:
            return PsTypedConstant(value, dtype)
        except PsTypeError:
            return None

    def __init__(self, value: Any, dtype: PsNumericType):
        """Create a new `PsTypedConstant`.

        The constructor of `PsTypedConstant` will first convert the given `dtype` to its const version.
        The given `value` will then be interpreted as that data type. The constructor will fail with an
        exception if that is not possible.
        """
        if not isinstance(dtype, PsNumericType):
            raise ValueError(f"Cannot create constant of type {dtype}")

        self._dtype = constify(dtype)
        self._value = self._dtype.create_constant(value)

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"PsTypedConstant( {self._value}, {repr(self._dtype)} )"

    @staticmethod
    def _fix(v: Any, dtype: PsNumericType) -> PsTypedConstant:
        if not isinstance(v, PsTypedConstant):
            return PsTypedConstant(v, dtype)
        else:
            return v

    @staticmethod
    def _bin_op(
        lhs: PsTypedConstant, rhs: PsTypedConstant, op: Callable[[Any, Any], Any]
    ) -> PsTypedConstant:
        """Backend for binary operators. Never call directly!"""

        if lhs._dtype != rhs._dtype:
            raise PsTypeError(
                f"Incompatible operand types in constant folding: {lhs._dtype} and {rhs._dtype}"
            )

        try:
            return PsTypedConstant(op(lhs._value, rhs._value), lhs._dtype)
        except PsTypeError:
            raise PsTypeError(
                f"Invalid operation in constant folding: {op.__name__}( {repr(lhs)}, {repr(rhs)}  )"
            )

    def __add__(self, other: Any):
        return PsTypedConstant._bin_op(
            self, PsTypedConstant._fix(other, self._dtype), operator.add
        )

    def __radd__(self, other: Any):
        return PsTypedConstant._bin_op(
            PsTypedConstant._fix(other, self._dtype), self, operator.add
        )

    def __mul__(self, other: Any):
        return PsTypedConstant._bin_op(
            self, PsTypedConstant._fix(other, self._dtype), operator.mul
        )

    def __rmul__(self, other: Any):
        return PsTypedConstant._bin_op(
            PsTypedConstant._fix(other, self._dtype), self, operator.mul
        )

    def __sub__(self, other: Any):
        return PsTypedConstant._bin_op(
            self, PsTypedConstant._fix(other, self._dtype), operator.sub
        )

    def __rsub__(self, other: Any):
        return PsTypedConstant._bin_op(
            PsTypedConstant._fix(other, self._dtype), self, operator.sub
        )

    def __truediv__(self, other: Any):
        other2 = PsTypedConstant._fix(other, self._dtype)
        if self._dtype.is_float():
            return PsTypedConstant._bin_op(self, other2, operator.truediv)
        else:
            return NotImplemented  # todo: C integer division

    def __rtruediv__(self, other: Any):
        other2 = PsTypedConstant._fix(other, self._dtype)
        if self._dtype.is_float():
            return PsTypedConstant._bin_op(other2, self, operator.truediv)
        else:
            return NotImplemented  # todo: C integer division

    def __mod__(self, other: Any):
        return NotImplemented  # todo: C integer division

    def __neg__(self):
        return PsTypedConstant(-self._value, self._dtype)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsTypedConstant):
            return False

        return self._dtype == other._dtype and self._value == other._value

    def __hash__(self) -> int:
        return hash((self._value, self._dtype))


pb.VALID_CONSTANT_CLASSES += (PsTypedConstant,)
