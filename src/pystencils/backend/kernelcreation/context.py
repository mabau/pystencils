from __future__ import annotations

from typing import Iterable, Iterator
from itertools import chain
from types import EllipsisType
from collections import namedtuple

from ...defaults import DEFAULTS
from ...field import Field, FieldType
from ...sympyextensions.typed_sympy import TypedSymbol

from ..symbols import PsSymbol
from ..arrays import PsLinearizedArray
from ...types import PsType, PsIntegerType, PsNumericType, PsScalarType, PsStructType
from ..constraints import KernelParamsConstraint
from ..exceptions import PsInternalCompilerError, KernelConstraintsError

from .iteration_space import IterationSpace, FullIterationSpace, SparseIterationSpace


class FieldsInKernel:
    def __init__(self) -> None:
        self.domain_fields: set[Field] = set()
        self.index_fields: set[Field] = set()
        self.custom_fields: set[Field] = set()
        self.buffer_fields: set[Field] = set()

    def __iter__(self) -> Iterator:
        return chain(
            self.domain_fields,
            self.index_fields,
            self.custom_fields,
            self.buffer_fields,
        )


FieldArrayPair = namedtuple("FieldArrayPair", ("field", "array"))


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

    def __init__(
        self,
        default_dtype: PsNumericType = DEFAULTS.numeric_dtype,
        index_dtype: PsIntegerType = DEFAULTS.index_dtype,
    ):
        self._default_dtype = default_dtype
        self._index_dtype = index_dtype

        self._symbols: dict[str, PsSymbol] = dict()

        self._fields_and_arrays: dict[str, FieldArrayPair] = dict()
        self._fields_collection = FieldsInKernel()

        self._ispace: IterationSpace | None = None

        self._constraints: list[KernelParamsConstraint] = []
        self._req_headers: set[str] = set()

    @property
    def default_dtype(self) -> PsNumericType:
        return self._default_dtype

    @property
    def index_dtype(self) -> PsIntegerType:
        return self._index_dtype

    #   Constraints

    def add_constraints(self, *constraints: KernelParamsConstraint):
        self._constraints += constraints

    @property
    def constraints(self) -> tuple[KernelParamsConstraint, ...]:
        return tuple(self._constraints)

    #   Symbols

    def get_symbol(self, name: str, dtype: PsType | None = None) -> PsSymbol:
        if name not in self._symbols:
            symb = PsSymbol(name, None)
            self._symbols[name] = symb
        else:
            symb = self._symbols[name]

        if dtype is not None:
            symb.apply_dtype(dtype)

        return symb

    def add_symbol(self, symbol: PsSymbol):
        if symbol.name in self._symbols:
            raise PsInternalCompilerError(f"Duplicate symbol: {symbol.name}")

        self._symbols[symbol.name] = symbol

    def replace_symbol(self, old: PsSymbol, new: PsSymbol):
        if old.name != new.name:
            raise PsInternalCompilerError(
                "replace_symbol: Old and new symbol must have the same name"
            )

        if old.name not in self._symbols:
            raise PsInternalCompilerError("Trying to replace an unknown symbol")

        self._symbols[old.name] = new

    @property
    def symbols(self) -> Iterable[PsSymbol]:
        return self._symbols.values()

    #   Fields and Arrays

    @property
    def fields(self) -> FieldsInKernel:
        return self._fields_collection

    def add_field(self, field: Field):
        """Add the given field to the context's fields collection.

        This method adds the passed ``field`` to the context's field collection, which is
        accesible through the `fields` member, and creates an array representation of the field,
        which is retrievable through `get_array`.
        Before adding the field to the collection, various sanity and constraint checks are applied.
        """

        if field in self._fields_and_arrays:
            #   Field was already added
            return

        arr_shape: list[EllipsisType | int] | None = None
        arr_strides: list[EllipsisType | int] | None = None

        #   Check field constraints and add to collection
        match field.field_type:
            case FieldType.GENERIC | FieldType.STAGGERED | FieldType.STAGGERED_FLUX:
                self._fields_collection.domain_fields.add(field)

            case FieldType.BUFFER:
                if field.spatial_dimensions != 1:
                    raise KernelConstraintsError(
                        f"Invalid spatial shape of buffer field {field.name}: {field.spatial_dimensions}. "
                        "Buffer fields must be one-dimensional."
                    )

                if field.index_dimensions > 1:
                    raise KernelConstraintsError(
                        f"Invalid index shape of buffer field {field.name}: {field.spatial_dimensions}. "
                        "Buffer fields can have at most one index dimension."
                    )

                num_entries = field.index_shape[0] if field.index_shape else 1
                if not isinstance(num_entries, int):
                    raise KernelConstraintsError(
                        f"Invalid index shape of buffer field {field.name}: {field.spatial_dimensions}. "
                        "Buffer fields cannot have variable index shape."
                    )

                arr_shape = [..., num_entries]
                arr_strides = [num_entries, 1]

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

        #   For non-buffer fields, determine shape and strides

        if arr_shape is None:
            arr_shape = [
                (
                    Ellipsis if isinstance(s, TypedSymbol) else s
                )  # TODO: Field should also use ellipsis
                for s in field.shape
            ]

            arr_strides = [
                (
                    Ellipsis if isinstance(s, TypedSymbol) else s
                )  # TODO: Field should also use ellipsis
                for s in field.strides
            ]

            # The frontend doesn't quite agree with itself on how to model
            # fields with trivial index dimensions. Sometimes the index_shape is empty,
            # sometimes its (1,). This is canonicalized here.
            if not field.index_shape:
                arr_shape += [1]
                arr_strides += [1]

        #   Add array
        assert arr_strides is not None

        assert isinstance(field.dtype, (PsScalarType, PsStructType))
        element_type = field.dtype

        arr = PsLinearizedArray(
            field.name, element_type, arr_shape, arr_strides, self.index_dtype
        )

        self._fields_and_arrays[field.name] = FieldArrayPair(field, arr)
        for symb in chain([arr.base_pointer], arr.shape, arr.strides):
            if isinstance(symb, PsSymbol):
                self.add_symbol(symb)

    @property
    def arrays(self) -> Iterable[PsLinearizedArray]:
        # return self._fields_and_arrays.values()
        yield from (item.array for item in self._fields_and_arrays.values())

    def get_array(self, field: Field) -> PsLinearizedArray:
        """Retrieve the underlying array for a given field.

        If the given field was not previously registered using `add_field`,
        this method internally calls `add_field` to check the field for consistency.
        """
        if field.name in self._fields_and_arrays:
            if field != self._fields_and_arrays[field.name].field:
                raise KernelConstraintsError(
                    "Encountered two fields of the same name but with different properties."
                )
        else:
            self.add_field(field)
        return self._fields_and_arrays[field.name].array

    def find_field(self, name: str) -> Field:
        return self._fields_and_arrays[name].field

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

    #   Headers

    @property
    def required_headers(self) -> set[str]:
        return self._req_headers

    def require_header(self, header: str):
        self._req_headers.add(header)
