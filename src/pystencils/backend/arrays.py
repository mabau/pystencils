from __future__ import annotations

from typing import Sequence
from types import EllipsisType

from abc import ABC

from .constants import PsConstant
from ..types import (
    PsType,
    PsPointerType,
    PsIntegerType,
    PsUnsignedIntegerType,
)

from .symbols import PsSymbol
from ..defaults import DEFAULTS


class PsLinearizedArray:
    """Class to model N-dimensional contiguous arrays.

    **Memory Layout, Shape and Strides**

    The memory layout of an array is defined by its shape and strides.
    Both shape and stride entries may either be constants or special variables associated with
    exactly one array.

    Shape and strides may be specified at construction in the following way.
    For constant entries, their value must be given as an integer.
    For variable shape entries and strides, the Ellipsis `...` must be passed instead.
    Internally, the passed ``index_dtype`` will be used to create typed constants (`PsConstant`)
    and variables (`PsArrayShapeSymbol` and `PsArrayStrideSymbol`) from the passed values.
    """

    def __init__(
        self,
        name: str,
        element_type: PsType,
        shape: Sequence[int | str | EllipsisType],
        strides: Sequence[int | str | EllipsisType],
        index_dtype: PsIntegerType = DEFAULTS.index_dtype,
    ):
        self._name = name
        self._element_type = element_type
        self._index_dtype = index_dtype

        if len(shape) != len(strides):
            raise ValueError("Shape and stride tuples must have the same length")

        def make_shape(coord, name_or_val):
            match name_or_val:
                case EllipsisType():
                    return PsArrayShapeSymbol(DEFAULTS.field_shape_name(name, coord), self, coord)
                case str():
                    return PsArrayShapeSymbol(name_or_val, self, coord)
                case _:
                    return PsConstant(name_or_val, index_dtype)

        self._shape: tuple[PsArrayShapeSymbol | PsConstant, ...] = tuple(
            make_shape(i, s) for i, s in enumerate(shape)
        )

        def make_stride(coord, name_or_val):
            match name_or_val:
                case EllipsisType():
                    return PsArrayStrideSymbol(DEFAULTS.field_stride_name(name, coord), self, coord)
                case str():
                    return PsArrayStrideSymbol(name_or_val, self, coord)
                case _:
                    return PsConstant(name_or_val, index_dtype)

        self._strides: tuple[PsArrayStrideSymbol | PsConstant, ...] = tuple(
            make_stride(i, s) for i, s in enumerate(strides)
        )

        self._base_ptr = PsArrayBasePointer(DEFAULTS.field_pointer_name(name), self)

    @property
    def name(self):
        """The array's name"""
        return self._name

    @property
    def base_pointer(self) -> PsArrayBasePointer:
        """The array's base pointer"""
        return self._base_ptr

    @property
    def shape(self) -> tuple[PsArrayShapeSymbol | PsConstant, ...]:
        """The array's shape, expressed using `PsConstant` and `PsArrayShapeSymbol`"""
        return self._shape

    @property
    def strides(self) -> tuple[PsArrayStrideSymbol | PsConstant, ...]:
        """The array's strides, expressed using `PsConstant` and `PsArrayStrideSymbol`"""
        return self._strides

    @property
    def index_type(self) -> PsIntegerType:
        return self._index_dtype

    @property
    def element_type(self) -> PsType:
        return self._element_type

    def __repr__(self) -> str:
        return (
            f"PsLinearizedArray({self._name}: {self.element_type}[{len(self.shape)}D])"
        )


class PsArrayAssocSymbol(PsSymbol, ABC):
    """A variable that is associated to an array.

    Instances of this class represent pointers and indexing information bound
    to a particular array.
    """

    __match_args__ = ("name", "dtype", "array")

    def __init__(self, name: str, dtype: PsType, array: PsLinearizedArray):
        super().__init__(name, dtype)
        self._array = array

    @property
    def array(self) -> PsLinearizedArray:
        return self._array


class PsArrayBasePointer(PsArrayAssocSymbol):
    def __init__(self, name: str, array: PsLinearizedArray):
        dtype = PsPointerType(array.element_type)
        super().__init__(name, dtype, array)

        self._array = array


class TypeErasedBasePointer(PsArrayBasePointer):
    """Base pointer for arrays whose element type has been erased.

    Used primarily for arrays of anonymous structs."""

    def __init__(self, name: str, array: PsLinearizedArray):
        dtype = PsPointerType(PsUnsignedIntegerType(8))
        super(PsArrayBasePointer, self).__init__(name, dtype, array)

        self._array = array


class PsArrayShapeSymbol(PsArrayAssocSymbol):
    """Variable that represents an array's shape in one coordinate.

    Do not instantiate this class yourself, but only use its instances
    as provided by `PsLinearizedArray.shape`.
    """

    __match_args__ = PsArrayAssocSymbol.__match_args__ + ("coordinate",)

    def __init__(
        self,
        name: str,
        array: PsLinearizedArray,
        coordinate: int,
    ):
        super().__init__(name, array.index_type, array)
        self._coordinate = coordinate

    @property
    def coordinate(self) -> int:
        return self._coordinate


class PsArrayStrideSymbol(PsArrayAssocSymbol):
    """Variable that represents an array's stride in one coordinate.

    Do not instantiate this class yourself, but only use its instances
    as provided by `PsLinearizedArray.strides`.
    """

    __match_args__ = PsArrayAssocSymbol.__match_args__ + ("coordinate",)

    def __init__(
        self,
        name: str,
        array: PsLinearizedArray,
        coordinate: int,
    ):
        super().__init__(name, array.index_type, array)
        self._coordinate = coordinate

    @property
    def coordinate(self) -> int:
        return self._coordinate
