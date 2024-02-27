"""
The pystencils backend models contiguous n-dimensional arrays using a number of classes.
Arrays themselves are represented through the `PsLinearizedArray` class.
An array has a fixed name, dimensionality, and element type, as well as a number of associated
variables.

The associated variables are the *shape* and *strides* of the array, modelled by the
`PsArrayShapeSymbol` and `PsArrayStrideSymbol` classes. They have integer type and are used to
reason about the array's memory layout.


Memory Layout Constraints
-------------------------

Initially, all memory layout information about an array is symbolic and unconstrained.
Several scenarios exist where memory layout must be constrained, e.g. certain pointers
need to be aligned, certain strides must be fixed or fulfill certain alignment properties,
or even the field shape must be fixed.

The code generation backend models such requirements and assumptions as *constraints*.
Constraints are external to the arrays themselves. They are created by the AST passes which
require them and exposed through the `PsKernelFunction` class to the compiler kernel's runtime
environment. It is the responsibility of the runtime environment to fulfill all constraints.

For example, if an array ``arr`` should have both a fixed shape and fixed strides,
an optimization pass will have to add equality constraints like the following before replacing
all occurences of the shape and stride variables with their constant value::

    constraints = (
        [PsKernelConstraint(s.eq(f)) for s, f in zip(arr.shape, fixed_size)] 
        + [PsKernelConstraint(s.eq(f)) for s, f in zip(arr.strides, fixed_strides)]
    )

    kernel_function.add_constraints(*constraints)

"""

from __future__ import annotations

from typing import Sequence
from types import EllipsisType

from abc import ABC

from .constants import PsConstant
from .types import (
    PsAbstractType,
    PsPointerType,
    PsIntegerType,
    PsUnsignedIntegerType,
    PsSignedIntegerType,
)

from .symbols import PsSymbol


class PsLinearizedArray:
    """Class to model N-dimensional contiguous arrays.

    Memory Layout, Shape and Strides
    --------------------------------

    The memory layout of an array is defined by its shape and strides.
    Both shape and stride entries may either be constants or special variables associated with
    exactly one array.

    Shape and strides may be specified at construction in the following way.
    For constant entries, their value must be given as an integer.
    For variable shape entries and strides, the Ellipsis `...` must be passed instead.
    Internally, the passed ``index_dtype`` will be used to create typed constants (`PsTypedConstant`)
    and variables (`PsArrayShapeSymbol` and `PsArrayStrideSymbol`) from the passed values.
    """

    def __init__(
        self,
        name: str,
        element_type: PsAbstractType,
        shape: Sequence[int | EllipsisType],
        strides: Sequence[int | EllipsisType],
        index_dtype: PsIntegerType = PsSignedIntegerType(64),
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
        """The array's shape, expressed using `PsTypedConstant` and `PsArrayShapeSymbol`"""
        return self._shape

    @property
    def shape_spec(self) -> tuple[EllipsisType | int, ...]:
        """The array's shape, expressed using `int` and `...`"""
        return tuple(
            (s.value if isinstance(s, PsConstant) else ...) for s in self._shape
        )

    @property
    def strides(self) -> tuple[PsArrayStrideSymbol | PsConstant, ...]:
        """The array's strides, expressed using `PsTypedConstant` and `PsArrayStrideSymbol`"""
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

    def __init__(self, name: str, dtype: PsAbstractType, array: PsLinearizedArray):
        super().__init__(name, dtype)
        self._array = array

    @property
    def array(self) -> PsLinearizedArray:
        return self._array


class PsArrayBasePointer(PsArrayAssocSymbol):
    __match_args__ = ("name", "array")

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

    __match_args__ = ("array", "coordinate", "dtype")

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
    __match_args__ = ("array", "coordinate", "dtype")

    def __init__(self, array: PsLinearizedArray, coordinate: int, dtype: PsIntegerType):
        name = f"{array.name}_stride{coordinate}"
        super().__init__(name, dtype, array)
        self._coordinate = coordinate

    @property
    def coordinate(self) -> int:
        return self._coordinate
