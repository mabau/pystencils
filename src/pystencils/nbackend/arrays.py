"""
Arrays
======

The pystencils backend models contiguous n-dimensional arrays using a number of classes.
Arrays themselves are represented through the `PsLinearizedArray` class.
An array has a fixed name, dimensionality, and element type, as well as a number of associated
variables.

The associated variables are the *shape* and *strides* of the array, modelled by the
`PsArrayShapeVar` and `PsArrayStrideVar` classes. They have integer type and are used to
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

For example, if an array `arr` should have both a fixed shape and fixed strides,
an optimization pass will have to add equality constraints like the following before replacing
all occurences of the shape and stride variables with their constant value:

```
constraints = (
    [PsKernelConstraint(s.eq(f)) for s, f in zip(arr.shape, fixed_size)] 
    + [PsKernelConstraint(s.eq(f)) for s, f in zip(arr.strides, fixed_strides)]
)

kernel_function.add_constraints(*constraints)
```

"""


from __future__ import annotations
from sys import intern

from types import EllipsisType

from abc import ABC

import pymbolic.primitives as pb

from .types import PsAbstractType, PsPointerType, PsIntegerType, PsSignedIntegerType

from .typed_expressions import PsTypedVariable, ExprOrConstant, PsTypedConstant


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
    Internally, the passed `index_dtype` will be used to create typed constants (`PsTypedConstant`)
    and variables (`PsArrayShapeVar` and `PsArrayStrideVar`) from the passed values.
    """

    def __init__(
        self,
        name: str,
        element_type: PsAbstractType,
        shape: tuple[int | EllipsisType, ...],
        strides: tuple[int | EllipsisType, ...],
        index_dtype: PsIntegerType = PsSignedIntegerType(64),
    ):
        self._name = name
        self._element_type = element_type
        self._index_dtype = index_dtype

        if len(shape) != len(strides):
            raise ValueError("Shape and stride tuples must have the same length")

        self._shape: tuple[PsArrayShapeVar | PsTypedConstant, ...] = tuple(
            (
                PsArrayShapeVar(self, i, index_dtype)
                if s == Ellipsis
                else PsTypedConstant(s, index_dtype)
            )
            for i, s in enumerate(shape)
        )

        self._strides: tuple[PsArrayStrideVar | PsTypedConstant, ...] = tuple(
            (
                PsArrayStrideVar(self, i, index_dtype)
                if s == Ellipsis
                else PsTypedConstant(s, index_dtype)
            )
            for i, s in enumerate(strides)
        )

        self._base_ptr = PsArrayBasePointer(f"{self._name}_data", self)

    @property
    def name(self):
        return self._name
    
    @property
    def base_pointer(self) -> PsArrayBasePointer:
        return self._base_ptr

    @property
    def shape(self) -> tuple[PsArrayShapeVar | PsTypedConstant, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[PsArrayStrideVar | PsTypedConstant, ...]:
        return self._strides

    @property
    def element_type(self):
        return self._element_type

    def _hashable_contents(self):
        """Contents by which to compare two instances of `PsLinearizedArray`.

        Since equality checks on shape and stride variables internally check equality of their associated arrays,
        if these variables would occur in here, an infinite recursion would follow.
        Hence they are filtered and replaced by the ellipsis.
        """
        shape_clean = tuple(
            (s if isinstance(s, PsTypedConstant) else ...) for s in self._shape
        )
        strides_clean = tuple(
            (s if isinstance(s, PsTypedConstant) else ...) for s in self._strides
        )
        return (
            self._name,
            self._element_type,
            self._index_dtype,
            shape_clean,
            strides_clean,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsLinearizedArray):
            return False

        return self._hashable_contents() == other._hashable_contents()

    def __hash__(self) -> int:
        return hash(self._hashable_contents())
    
    def __repr__(self) -> str:
        return f"PsLinearizedArray({self._name}: {self.element_type}[{len(self.shape)}D])"


class PsArrayAssocVar(PsTypedVariable, ABC):
    """A variable that is associated to an array.

    Instances of this class represent pointers and indexing information bound
    to a particular array.
    """

    init_arg_names: tuple[str, ...] = ("name", "dtype", "array")
    __match_args__ = ("name", "dtype", "array")

    def __init__(self, name: str, dtype: PsAbstractType, array: PsLinearizedArray):
        super().__init__(name, dtype)
        self._array = array

    def __getinitargs__(self):
        return self.name, self.dtype, self.array

    @property
    def array(self) -> PsLinearizedArray:
        return self._array


class PsArrayBasePointer(PsArrayAssocVar):
    init_arg_names: tuple[str, ...] = ("name", "array")
    __match_args__ = ("name", "array")

    def __init__(self, name: str, array: PsLinearizedArray):
        dtype = PsPointerType(array.element_type)
        super().__init__(name, dtype, array)

        self._array = array

    def __getinitargs__(self):
        return self.name, self.array


class PsArrayShapeVar(PsArrayAssocVar):
    """Variable that represents an array's shape in one coordinate.

    Do not instantiate this class yourself, but only use its instances
    as provided by `PsLinearizedArray.shape`.
    """

    init_arg_names: tuple[str, ...] = ("array", "coordinate", "dtype")
    __match_args__ = ("array", "coordinate", "dtype")

    def __init__(self, array: PsLinearizedArray, coordinate: int, dtype: PsIntegerType):
        name = f"{array.name}_size{coordinate}"
        super().__init__(name, dtype, array)
        self._coordinate = coordinate

    @property
    def coordinate(self) -> int:
        return self._coordinate

    def __getinitargs__(self):
        return self.array, self.coordinate, self.dtype


class PsArrayStrideVar(PsArrayAssocVar):
    """Variable that represents an array's stride in one coordinate.

    Do not instantiate this class yourself, but only use its instances
    as provided by `PsLinearizedArray.strides`.
    """

    init_arg_names: tuple[str, ...] = ("array", "coordinate", "dtype")
    __match_args__ = ("array", "coordinate", "dtype")

    def __init__(self, array: PsLinearizedArray, coordinate: int, dtype: PsIntegerType):
        name = f"{array.name}_size{coordinate}"
        super().__init__(name, dtype, array)
        self._coordinate = coordinate

    @property
    def coordinate(self) -> int:
        return self._coordinate

    def __getinitargs__(self):
        return self.array, self.coordinate, self.dtype


class PsArrayAccess(pb.Subscript):

    mapper_method = intern("map_array_access")

    def __init__(self, base_ptr: PsArrayBasePointer, index: ExprOrConstant):
        super(PsArrayAccess, self).__init__(base_ptr, index)
        self._base_ptr = base_ptr
        self._index = index

    @property
    def base_ptr(self):
        return self._base_ptr

    @property
    def array(self) -> PsLinearizedArray:
        return self._base_ptr.array

    @property
    def dtype(self) -> PsAbstractType:
        """Data type of this expression, i.e. the element type of the underlying array"""
        return self._base_ptr.array.element_type
