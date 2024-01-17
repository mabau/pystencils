from __future__ import annotations
from typing import cast
from dataclasses import dataclass

from abc import ABC

from ...field import Field
from ...typing import TypedSymbol, BasicType

from ..arrays import PsLinearizedArray, PsArrayBasePointer
from ..types import PsIntegerType
from ..types.quick import make_type
from ..typed_expressions import PsTypedVariable, VarOrConstant
from ..constraints import PsKernelConstraint


@dataclass
class PsFieldArrayPair:
    field: Field
    array: PsLinearizedArray
    base_ptr: PsArrayBasePointer


class IterationSpace(ABC):
    """Represents the n-dimensonal iteration space of a pystencils kernel.

    Instances of this class represent the kernel's iteration region during translation from
    SymPy, before any indexing sources are generated. It provides the counter symbols which
    should be used to translate field accesses to array accesses.

    There are two types of iteration spaces, modelled by subclasses:
     - The full iteration space translates to an n-dimensional loop nest or the corresponding device
       indexing scheme.
     - The sparse iteration space translates to a single loop over an index list which in turn provides the
       spatial indices.
    """

    def __init__(self, spatial_indices: tuple[PsTypedVariable, ...]):
        if len(spatial_indices) == 0:
            raise ValueError("Iteration space must be at least one-dimensional.")

        self._spatial_indices = spatial_indices

    @property
    def spatial_indices(self) -> tuple[PsTypedVariable, ...]:
        return self._spatial_indices


class FullIterationSpace(IterationSpace):
    def __init__(
        self,
        lower: tuple[VarOrConstant, ...],
        upper: tuple[VarOrConstant, ...],
        counters: tuple[PsTypedVariable, ...],
    ):
        if not (len(lower) == len(upper) == len(counters)):
            raise ValueError(
                "Lower and upper iteration limits and counters must have the same shape."
            )

        super().__init__(counters)

        self._lower = lower
        self._upper = upper
        self._counters = counters

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper


class SparseIterationSpace(IterationSpace):
    def __init__(self, spatial_index_variables: tuple[PsTypedVariable, ...]):
        super().__init__(spatial_index_variables)
        # todo


class KernelCreationContext:
    """Manages the translation process from the SymPy frontend to the backend AST.

    It does the following things:

      - Default data types: The context knows the data types that should be applied by default
        to SymPy expressions.
      - Management of fields. The context manages all mappings from front-end `Field`s to their
        underlying `PsLinearizedArray`s.
      - Collection of constraints. All constraints that arise during translation are collected in the
        context, and finally attached to the kernel function object once translation is complete.

    Data Types
    ----------

     - The `index_dtype` is the data type used throughout translation for all loop counters and array indexing.
     - The `default_numeric_dtype` is the data type assigned by default to all symbols occuring in SymPy assignments

    Fields and Arrays
    -----------------

    There's several types of fields that need to be mapped to arrays.

    - `FieldType.GENERIC` corresponds to domain fields.
      Domain fields can only be accessed by relative offsets, and therefore must always
      be associated with an iteration space that provides a spatial index tuple.
    - `FieldType.INDEXED` are 1D arrays of index structures. They must be accessed by a single running index.
      If there is at least one indexed field present there must also exist an index source for that field
      (loop or device indexing).
      An indexed field may itself be an index source for domain fields.
    - `FieldType.BUFFER` are 1D arrays whose indices must be incremented with each access.
      Within a domain, a buffer may be either written to or read from, never both.


    In the translator, frontend fields and backend arrays are managed together using the `PsFieldArrayPair` class.
    """

    def __init__(self, index_dtype: PsIntegerType):
        self._index_dtype = index_dtype
        self._arrays: dict[Field, PsFieldArrayPair] = dict()
        self._constraints: list[PsKernelConstraint] = []

    @property
    def index_dtype(self) -> PsIntegerType:
        return self._index_dtype

    def add_constraints(self, *constraints: PsKernelConstraint):
        self._constraints += constraints

    @property
    def constraints(self) -> tuple[PsKernelConstraint, ...]:
        return tuple(self._constraints)

    def add_field(self, field: Field) -> PsFieldArrayPair:
        arr_shape = tuple(
            (
                Ellipsis if isinstance(s, TypedSymbol) else s
            )  # TODO: Field should also use ellipsis
            for s in field.shape
        )

        arr_strides = tuple(
            (
                Ellipsis if isinstance(s, TypedSymbol) else s
            )  # TODO: Field should also use ellipsis
            for s in field.strides
        )

        # TODO: frontend should use new type system
        element_type = make_type(cast(BasicType, field.dtype).numpy_dtype.type)

        arr = PsLinearizedArray(
            field.name, element_type, arr_shape, arr_strides, self.index_dtype
        )

        fa_pair = PsFieldArrayPair(
            field=field, array=arr, base_ptr=PsArrayBasePointer("arr_data", arr)
        )

        self._arrays[field] = fa_pair

        return fa_pair

    def get_array_descriptor(self, field: Field):
        return self._arrays[field]
