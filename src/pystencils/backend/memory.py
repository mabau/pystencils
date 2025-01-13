from __future__ import annotations
from typing import Sequence
from itertools import chain
from dataclasses import dataclass

from ..types import PsType, PsTypeError, deconstify, PsIntegerType, PsPointerType
from .exceptions import PsInternalCompilerError
from .constants import PsConstant
from ..codegen.properties import PsSymbolProperty, UniqueSymbolProperty


class PsSymbol:
    """A mutable symbol with name and data type.

    Do not create objects of this class directly unless you know what you are doing;
    instead obtain them from a `KernelCreationContext` through `KernelCreationContext.get_symbol`.
    This way, the context can keep track of all symbols used in the translation run,
    and uniqueness of symbols is ensured.
    """

    __match_args__ = ("name", "dtype")

    def __init__(self, name: str, dtype: PsType | None = None):
        self._name = name
        self._dtype = dtype
        self._properties: set[PsSymbolProperty] = set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> PsType | None:
        return self._dtype

    @dtype.setter
    def dtype(self, value: PsType):
        self._dtype = value

    def apply_dtype(self, dtype: PsType):
        """Apply the given data type to this symbol,
        raising a TypeError if it conflicts with a previously set data type."""

        if self._dtype is not None and self._dtype != dtype:
            raise PsTypeError(
                f"Incompatible symbol data types: {self._dtype} and {dtype}"
            )

        self._dtype = dtype

    def get_dtype(self) -> PsType:
        if self._dtype is None:
            raise PsInternalCompilerError(
                f"Symbol {self.name} had no type assigned yet"
            )
        return self._dtype

    @property
    def properties(self) -> frozenset[PsSymbolProperty]:
        """Set of properties attached to this symbol"""
        return frozenset(self._properties)

    def get_properties(
        self, prop_type: type[PsSymbolProperty]
    ) -> set[PsSymbolProperty]:
        """Retrieve all properties of the given type attached to this symbol"""
        return set(filter(lambda p: isinstance(p, prop_type), self._properties))

    def add_property(self, property: PsSymbolProperty):
        """Attach a property to this symbol"""
        if isinstance(property, UniqueSymbolProperty) and not self.get_properties(
            type(property)
        ) <= {property}:
            raise ValueError(
                f"Cannot add second instance of unique property {type(property)} to symbol {self._name}."
            )

        self._properties.add(property)

    def remove_property(self, property: PsSymbolProperty):
        """Remove a property from this symbol. Does nothing if the property is not attached."""
        self._properties.discard(property)

    def __str__(self) -> str:
        dtype_str = "<untyped>" if self._dtype is None else str(self._dtype)
        return f"{self._name}: {dtype_str}"

    def __repr__(self) -> str:
        return f"PsSymbol({repr(self._name)}, {repr(self._dtype)})"


@dataclass(frozen=True)
class BufferBasePtr(UniqueSymbolProperty):
    """Symbol acts as a base pointer to a buffer."""

    buffer: PsBuffer


class PsBuffer:
    """N-dimensional contiguous linearized buffer in heap memory.

    `PsBuffer` models the memory buffers underlying the `Field` class
    to the backend. Each buffer represents a contiguous block of memory
    that is non-aliased and disjoint from all other buffers.

    Buffer shape and stride information are given either as constants or as symbols.
    All indexing expressions must have the same data type, which will be selected as the buffer's
    ``index_dtype <PsBuffer.index_dtype>``.

    Each buffer has at least one base pointer, which can be retrieved via the `PsBuffer.base_pointer`
    property.
    """

    def __init__(
        self,
        name: str,
        element_type: PsType,
        base_ptr: PsSymbol,
        shape: Sequence[PsSymbol | PsConstant],
        strides: Sequence[PsSymbol | PsConstant],
    ):
        bptr_type = base_ptr.get_dtype()
        
        if not isinstance(bptr_type, PsPointerType):
            raise ValueError(
                f"Type of buffer base pointer {base_ptr} was not a pointer type: {bptr_type}"
            )
        
        if bptr_type.base_type != element_type:
            raise ValueError(
                f"Base type of primary buffer base pointer {base_ptr} "
                f"did not equal buffer element type {element_type}."
            )

        if len(shape) != len(strides):
            raise ValueError("Buffer shape and stride tuples must have the same length")

        idx_types: set[PsType] = set(
            deconstify(s.get_dtype()) for s in chain(shape, strides)
        )
        if len(idx_types) > 1:
            raise ValueError(
                f"Conflicting data types in indexing symbols to buffer {name}: {idx_types}"
            )

        idx_dtype = idx_types.pop()
        if not isinstance(idx_dtype, PsIntegerType):
            raise ValueError(
                f"Invalid index data type for buffer {name}: {idx_dtype}. Must be an integer type."
            )

        self._name = name
        self._element_type = element_type
        self._index_dtype = idx_dtype

        self._shape = list(shape)
        self._strides = list(strides)

        base_ptr.add_property(BufferBasePtr(self))
        self._base_ptr = base_ptr

    @property
    def name(self):
        """The buffer's name"""
        return self._name

    @property
    def base_pointer(self) -> PsSymbol:
        """Primary base pointer"""
        return self._base_ptr

    @property
    def shape(self) -> list[PsSymbol | PsConstant]:
        """Buffer shape symbols and/or constants"""
        return self._shape

    @property
    def strides(self) -> list[PsSymbol | PsConstant]:
        """Buffer stride symbols and/or constants"""
        return self._strides

    @property
    def dim(self) -> int:
        """Dimensionality of this buffer"""
        return len(self._shape)

    @property
    def index_type(self) -> PsIntegerType:
        """Index data type of this buffer; i.e. data type of its shape and stride symbols"""
        return self._index_dtype

    @property
    def element_type(self) -> PsType:
        """Element type of this buffer"""
        return self._element_type

    def __repr__(self) -> str:
        return f"PsBuffer({self._name}: {self.element_type}[{len(self.shape)}D])"
