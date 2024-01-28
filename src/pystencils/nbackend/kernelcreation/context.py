from __future__ import annotations

from ...field import Field, FieldType
from ...typing import TypedSymbol, BasicType, StructType

from ..arrays import PsLinearizedArray
from ..types import PsIntegerType
from ..types.quick import make_type
from ..constraints import PsKernelConstraint
from ..exceptions import PsInternalCompilerError, KernelConstraintsError

from .options import KernelCreationOptions
from .iteration_space import IterationSpace, FullIterationSpace, SparseIterationSpace


class FieldsInKernel:
    def __init__(self) -> None:
        self.domain_fields: set[Field] = set()
        self.index_fields: set[Field] = set()
        self.custom_fields: set[Field] = set()
        self.buffer_fields: set[Field] = set()


class KernelCreationContext:
    """Manages the translation process from the SymPy frontend to the backend AST, and collects
    all necessary information for the translation.


    Data Types
    ----------

    The kernel creation context manages the default data types for loop limits and counters, index calculations,
    and the typifier.

    Fields and Arrays
    ------------------

    The kernel creation context acts as a factory for mapping fields to arrays.

    Iteration Space
    ---------------

    The context manages the iteration space within which the current translation takes place. It may be a sparse
    or full iteration space.
    """

    def __init__(self, options: KernelCreationOptions):
        self._options = options
        self._arrays: dict[Field, PsLinearizedArray] = dict()
        self._constraints: list[PsKernelConstraint] = []

        self._fields_collection = FieldsInKernel()
        self._ispace: IterationSpace | None = None

    @property
    def options(self) -> KernelCreationOptions:
        return self._options

    @property
    def index_dtype(self) -> PsIntegerType:
        return self._options.index_dtype

    def add_constraints(self, *constraints: PsKernelConstraint):
        self._constraints += constraints

    @property
    def constraints(self) -> tuple[PsKernelConstraint, ...]:
        return tuple(self._constraints)

    #   Fields and Arrays

    @property
    def fields(self) -> FieldsInKernel:
        return self._fields_collection

    def add_field(self, field: Field):
        """Add the given field to the context's fields collection"""
        match field.field_type:
            case FieldType.GENERIC | FieldType.STAGGERED | FieldType.STAGGERED_FLUX:
                self._fields_collection.domain_fields.add(field)

            case FieldType.BUFFER:
                if field.spatial_dimensions != 1:
                    raise KernelConstraintsError(
                        f"Invalid spatial shape of buffer field {field.name}: {field.spatial_dimensions}. "
                        "Buffer fields must be one-dimensional."
                    )
                self._fields_collection.buffer_fields.add(field)

            case FieldType.INDEXED:
                if field.spatial_dimensions != 1:
                    raise KernelConstraintsError(
                        f"Invalid spatial shape of index field {field.name}: {field.spatial_dimensions}. "
                        "Index fields must be one-dimensional."
                    )
                self._fields_collection.index_fields.add(field)

            case FieldType.CUSTOM:
                self._fields_collection.custom_fields.add(field)

            case _:
                assert False, "unreachable code"

    def get_array(self, field: Field) -> PsLinearizedArray:
        if field not in self._arrays:
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

            assert isinstance(field.dtype, (BasicType, StructType))
            element_type = make_type(field.dtype.numpy_dtype)

            arr = PsLinearizedArray(
                field.name, element_type, arr_shape, arr_strides, self.index_dtype
            )

            self._arrays[field] = arr

        return self._arrays[field]

    #   Iteration Space

    def set_iteration_space(self, ispace: IterationSpace):
        self._ispace = ispace

    def get_iteration_space(self) -> IterationSpace:
        if self._ispace is None:
            raise PsInternalCompilerError("No iteration space set in context.")
        return self._ispace

    def get_full_iteration_space(self) -> FullIterationSpace:
        if not isinstance(self._ispace, FullIterationSpace):
            raise PsInternalCompilerError("No full iteration space set in context.")
        return self._ispace

    def get_sparse_iteration_space(self) -> SparseIterationSpace:
        if not isinstance(self._ispace, SparseIterationSpace):
            raise PsInternalCompilerError("No sparse iteration space set in context.")
        return self._ispace
