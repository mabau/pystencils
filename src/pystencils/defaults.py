from typing import TypeVar, Generic, Callable
from .backend.types import PsAbstractType, PsIeeeFloatType, PsSignedIntegerType, PsStructType

from pystencils.sympyextensions.typed_sympy import TypedSymbol

SymbolT = TypeVar("SymbolT")


class GenericDefaults(Generic[SymbolT]):
    def __init__(self, symcreate: Callable[[str, PsAbstractType], SymbolT]):
        self.numeric_dtype = PsIeeeFloatType(64)
        """Default data type for numerical computations"""

        self.index_dtype = PsSignedIntegerType(64)
        """Default data type for indices."""

        self.spatial_counter_names = ("ctr_0", "ctr_1", "ctr_2")
        """Names of the default spatial counters"""

        self.spatial_counters = (
            symcreate("ctr_0", self.index_dtype),
            symcreate("ctr_1", self.index_dtype),
            symcreate("ctr_2", self.index_dtype),
        )
        """Default spatial counters"""

        self._index_struct_coordinate_names = ("x", "y", "z")
        """Default names of spatial coordinate members in index list structures"""

        self.index_struct_coordinates = (
            PsStructType.Member("x", self.index_dtype),
            PsStructType.Member("y", self.index_dtype),
            PsStructType.Member("z", self.index_dtype),
        )
        """Default spatial coordinate members in index list structures"""

        self.sparse_counter_name = "sparse_idx"
        """Name of the default sparse iteration counter"""

        self.sparse_counter = symcreate(self.sparse_counter_name, self.index_dtype)
        """Default sparse iteration counter."""


DEFAULTS = GenericDefaults[TypedSymbol](TypedSymbol)
"""Default names and symbols used throughout code generation"""
