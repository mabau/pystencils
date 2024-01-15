from __future__ import annotations

from functools import reduce
from typing import TypeAlias, Union, Any, Tuple

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
        length = reduce(lambda x, y: x * y, shape)
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


class PsTypedConstant:
    """Represents typed constants occuring in the pystencils AST.

    Internal Representation of Constants
    ------------------------------------

    Each `PsNumericType` acts as a factory for the code generator's internal representation of that type's
    constants. The `PsTypedConstant` class embedds these into the expression trees.
    Upon construction, this class's constructor attempts to interpret the given value in the given data type
    by passing it to the data type's factory, which in turn may throw an exception if the value's type does
    not match.

    Operations and Constant Folding
    -------------------------------

    The `PsTypedConstant` class overrides the basic arithmetic operations for use during a constant folding pass.
    Their implementations are very strict regarding types: No implicit conversions take place, and both operands
    must always have the exact same type.
    The only exception to this rule are the values `0`, `1`, and `-1`, which are promoted to `PsTypedConstant`
    (pymbolic injects those at times).

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

    def _fix(self, v: Any) -> PsTypedConstant:
        """In binary operations, checks for type equality and, if necessary, promotes the values
        `0`, `1` and `-1` to `PsTypedConstant`."""
        if not isinstance(v, PsTypedConstant) and v in (0, 1, -1):
            return PsTypedConstant(v, self._dtype)
        elif v._dtype != self._dtype:
            raise PsTypeError(
                f"Incompatible operand types in constant folding: {self._dtype} and {v._dtype}"
            )
        else:
            return v

    def _rfix(self, v: Any) -> PsTypedConstant:
        """Same as `_fix`, but for use with the `r...` versions of the binary ops. Only changes the order of the
        types in the exception string."""
        if not isinstance(v, PsTypedConstant) and v in (0, 1, -1):
            return PsTypedConstant(v, self._dtype)
        elif v._dtype != self._dtype:
            raise PsTypeError(
                f"Incompatible operand types in constant folding: {v._dtype} and {self._dtype}"
            )
        else:
            return v

    def __add__(self, other: Any):
        return PsTypedConstant(self._value + self._fix(other)._value, self._dtype)

    def __radd__(self, other: Any):
        return PsTypedConstant(self._rfix(other)._value + self._value, self._dtype)

    def __mul__(self, other: Any):
        return PsTypedConstant(self._value * self._fix(other)._value, self._dtype)

    def __rmul__(self, other: Any):
        return PsTypedConstant(self._rfix(other)._value * self._value, self._dtype)

    def __sub__(self, other: Any):
        return PsTypedConstant(self._value - self._fix(other)._value, self._dtype)

    def __rsub__(self, other: Any):
        return PsTypedConstant(self._rfix(other)._value - self._value, self._dtype)
    
    @staticmethod
    def _divrem(dividend, divisor):
        quotient =  abs(dividend) // abs(divisor)
        quotient = quotient if (dividend * divisor > 0) else (- quotient)
        rem = abs(dividend) % abs(divisor)
        rem = rem if dividend >= 0 else (- rem)
        return quotient, rem

    def __truediv__(self, other: Any):
        if self._dtype.is_float():
            return PsTypedConstant(self._value / self._fix(other)._value, self._dtype)
        elif self._dtype.is_uint():
            #   For unsigned integers, `//` does the correct thing
            return PsTypedConstant(self._value // self._fix(other)._value, self._dtype)
        elif self._dtype.is_sint():
            dividend = self._value
            divisor = self._fix(other)._value
            quotient, _ = self._divrem(dividend, divisor)
            return PsTypedConstant(quotient, self._dtype)
        else:
            return NotImplemented

    def __rtruediv__(self, other: Any):
        if self._dtype.is_float():
            return PsTypedConstant(self._rfix(other)._value / self._value, self._dtype)
        elif self._dtype.is_uint():
            return PsTypedConstant(self._rfix(other)._value // self._value, self._dtype)
        elif self._dtype.is_sint():
            dividend = self._fix(other)._value
            divisor = self._value
            quotient, _ = self._divrem(dividend, divisor)
            return PsTypedConstant(quotient, self._dtype)
        else:
            return NotImplemented

    def __mod__(self, other: Any):
        if self._dtype.is_uint():
            return PsTypedConstant(self._value % self._fix(other)._value, self._dtype)
        else:
            dividend = self._value
            divisor = self._fix(other)._value
            _, rem = self._divrem(dividend, divisor)
            return PsTypedConstant(rem, self._dtype)

    def __neg__(self):
        minus_one = PsTypedConstant(-1, self._dtype)
        return pb.Product((minus_one, self))
    
    def __bool__(self):
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsTypedConstant):
            return False

        return self._dtype == other._dtype and self._value == other._value

    def __hash__(self) -> int:
        return hash((self._value, self._dtype))


pb.register_constant_class(PsTypedConstant)


PsLvalue: TypeAlias = Union[PsTypedVariable, PsArrayAccess]
"""Types of expressions that may occur on the left-hand side of assignments."""

ExprOrConstant: TypeAlias = pb.Expression | PsTypedConstant
"""Required since `PsTypedConstant` does not derive from `pb.Expression`."""
