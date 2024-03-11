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
        shape: Sequence[int | EllipsisType],
        strides: Sequence[int | EllipsisType],
        index_dtype: PsIntegerType = DEFAULTS.index_dtype,
    ):
        self._name = name
        self._element_type = element_type
        self._index_dtype = index_dtype

        if len(shape) != len(strides):
            raise ValueError("Shape and stride tuples must have the same length")

        self._shape: tuple[PsArrayShapeSymbol | PsConstant, ...] = tuple(
            (
                PsArrayShapeSymbol(self, i, index_dtype)
                if s == Ellipsis
                else PsConstant(s, index_dtype)
            )
            for i, s in enumerate(shape)
        )

        self._strides: tuple[PsArrayStrideSymbol | PsConstant, ...] = tuple(
            (
                PsArrayStrideSymbol(self, i, index_dtype)
                if s == Ellipsis
                else PsConstant(s, index_dtype)
            )
            for i, s in enumerate(strides)
        )

        self._base_ptr = PsArrayBasePointer(f"{self._name}_data", self)

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
    def shape_spec(self) -> tuple[EllipsisType | int, ...]:
        """The array's shape, expressed using `int` and `...`"""
        return tuple(
            (s.value if isinstance(s, PsConstant) else ...) for s in self._shape
        )

    @property
    def strides(self) -> tuple[PsArrayStrideSymbol | PsConstant, ...]:
        """The array's strides, expressed using `PsConstant` and `PsArrayStrideSymbol`"""
        return self._strides

    @property
    def strides_spec(self) -> tuple[EllipsisType | int, ...]:
        """The array's strides, expressed using `int` and `...`"""
        return tuple(
            (s.value if isinstance(s, PsConstant) else ...) for s in self._strides
        )

    @property
    def element_type(self):
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

    def __init__(self, array: PsLinearizedArray, coordinate: int, dtype: PsIntegerType):
        name = f"{array.name}_size{coordinate}"
        super().__init__(name, dtype, array)
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

    def __init__(self, array: PsLinearizedArray, coordinate: int, dtype: PsIntegerType):
        name = f"{array.name}_stride{coordinate}"
        super().__init__(name, dtype, array)
        self._coordinate = coordinate

    @property
    def coordinate(self) -> int:
        return self._coordinate
