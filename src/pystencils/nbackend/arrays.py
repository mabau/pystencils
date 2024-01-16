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
    [PsParamConstraint(s.eq(f)) for s, f in zip(arr.shape, fixed_size)] 
    + [PsParamConstraint(s.eq(f)) for s, f in zip(arr.strides, fixed_strides)]
)

kernel_function.add_constraints(*constraints)
```

"""


from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC

import pymbolic.primitives as pb

from .types import (
    PsAbstractType,
    PsScalarType,
    PsPointerType,
    PsIntegerType,
    PsSignedIntegerType,
    constify,
)

if TYPE_CHECKING:
    from .typed_expressions import PsTypedVariable, PsTypedConstant


class PsLinearizedArray:
    """N-dimensional contiguous array"""

    def __init__(
        self,
        name: str,
        element_type: PsScalarType,
        dim: int,
        offsets: tuple[int, ...] | None = None,
        index_dtype: PsIntegerType = PsSignedIntegerType(64),
    ):
        self._name = name

        if offsets is not None and len(offsets) != dim:
            raise ValueError(f"Must have exactly {dim} offsets.")

        self._shape = tuple(
            PsArrayShapeVar(self, d, constify(index_dtype)) for d in range(dim)
        )
        self._strides = tuple(
            PsArrayStrideVar(self, d, constify(index_dtype)) for d in range(dim)
        )
        self._element_type = element_type

        if offsets is None:
            offsets = (0,) * dim

        self._offsets = tuple(PsTypedConstant(o, index_dtype) for o in offsets)

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def element_type(self):
        return self._element_type

    @property
    def offsets(self) -> tuple[PsTypedConstant, ...]:
        return self._offsets


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
    init_arg_names: tuple[str, ...] = ("array", "coordinate", "dtype")
    __match_args__ = ("array", "coordinate", "dtype")

    def __init__(self, array: PsLinearizedArray, coordinate: int, dtype: PsIntegerType):
        name = f"{array}_size{coordinate}"
        super().__init__(name, dtype, array)
        self._coordinate = coordinate

    @property
    def coordinate(self) -> int:
        return self._coordinate

    def __getinitargs__(self):
        return self.array, self.coordinate, self.dtype


class PsArrayStrideVar(PsArrayAssocVar):
    init_arg_names: tuple[str, ...] = ("array", "coordinate", "dtype")
    __match_args__ = ("array", "coordinate", "dtype")

    def __init__(self, array: PsLinearizedArray, coordinate: int, dtype: PsIntegerType):
        name = f"{array}_size{coordinate}"
        super().__init__(name, dtype, array)
        self._coordinate = coordinate

    @property
    def coordinate(self) -> int:
        return self._coordinate

    def __getinitargs__(self):
        return self.array, self.coordinate, self.dtype


class PsArrayAccess(pb.Subscript):
    def __init__(self, base_ptr: PsArrayBasePointer, index: pb.Expression):
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


# class PsIterationDomain:
#     """A factory for arrays spanning a given iteration domain."""

#     def __init__(
#         self,
#         id: str,
#         dim: int | None = None,
#         fixed_shape: tuple[int, ...] | None = None,
#         index_dtype: PsIntegerType = PsSignedIntegerType(64),
#     ):
#         if fixed_shape is not None:
#             if dim is not None and len(fixed_shape) != dim:
#                 raise ValueError(
#                     "If both `dim` and `fixed_shape` are specified, `fixed_shape` must have exactly `dim` entries."
#                 )

#             shape = tuple(PsTypedConstant(s, index_dtype) for s in fixed_shape)
#         elif dim is not None:
#             shape = tuple(
#                 PsTypedVariable(f"{id}_shape_{d}", index_dtype) for d in range(dim)
#             )
#         else:
#             raise ValueError("Either `fixed_shape` or `dim` must be specified.")

#         self._domain_shape: tuple[VarOrConstant, ...] = shape
#         self._index_dtype = index_dtype

#         self._archetype_array: PsLinearizedArray | None = None

#         self._constraints: list[PsParamConstraint] = []

#     @property
#     def dim(self) -> int:
#         return len(self._domain_shape)

#     @property
#     def shape(self) -> tuple[VarOrConstant, ...]:
#         return self._domain_shape

#     def create_array(self, ghost_layers: int = 0):
