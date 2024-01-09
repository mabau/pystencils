from __future__ import annotations

from functools import reduce
from typing import TypeAlias, Union, Any, Tuple

import pymbolic.primitives as pb

from ..typing import AbstractType, BasicType, PointerType


class PsTypedVariable(pb.Variable):
    def __init__(self, name: str, dtype: AbstractType):
        super(PsTypedVariable, self).__init__(name)
        self._dtype = dtype

    @property
    def dtype(self) -> AbstractType:
        return self._dtype


class PsArray:
    def __init__(
        self,
        name: str,
        length: pb.Expression,
        element_type: BasicType,  # todo Frederik: is BasicType correct?
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
        element_type: BasicType,
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
        dtype = PointerType(array.element_type)
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

    @property
    def index(self):
        return self._index

    @property
    def array(self) -> PsArray:
        return self._base_ptr.array
    
    @property
    def dtype(self) -> AbstractType:
        """Data type of this expression, i.e. the element type of the underlying array"""
        return self._base_ptr.array.element_type


PsLvalue: TypeAlias = Union[PsTypedVariable, PsArrayAccess]


class PsTypedConstant:
    @staticmethod
    def _cast(value, target_dtype: AbstractType):
        if isinstance(value, PsTypedConstant):
            if value._dtype != target_dtype:
                raise ValueError(
                    f"Incompatible types: {value._dtype} and {target_dtype}"
                )
            return value

        # TODO check legality
        return PsTypedConstant(value, target_dtype)

    def __init__(self, value, dtype: AbstractType):
        """Represents typed constants occuring in the pystencils AST"""
        if isinstance(dtype, BasicType):
            dtype = BasicType(dtype, const=True)
            self._value = dtype.numpy_dtype.type(value)
        else:
            raise ValueError(f"Cannot create constant of type {dtype}")

        self._dtype = dtype

    def __str__(self) -> str:
        return str(self._value)

    def __add__(self, other: Any):
        other = PsTypedConstant._cast(other, self._dtype)

        return PsTypedConstant(self._value + other._value, self._dtype)

    def __mul__(self, other: Any):
        other = PsTypedConstant._cast(other, self._dtype)

        return PsTypedConstant(self._value * other._value, self._dtype)

    def __sub__(self, other: Any):
        other = PsTypedConstant._cast(other, self._dtype)

        return PsTypedConstant(self._value - other._value, self._dtype)

    # TODO: Remaining operators


pb.VALID_CONSTANT_CLASSES += (PsTypedConstant,)
