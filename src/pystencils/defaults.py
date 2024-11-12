from .types import (
    PsIeeeFloatType,
    PsIntegerType,
    PsSignedIntegerType,
    PsStructType,
    UserTypeSpec,
    create_type,
)

from pystencils.sympyextensions.typed_sympy import TypedSymbol, DynamicType


class SympyDefaults:
    def __init__(self):
        self.numeric_dtype = PsIeeeFloatType(64)
        """Default data type for numerical computations"""

        self.index_dtype: PsIntegerType = PsSignedIntegerType(64)
        """Default data type for indices."""

        self.spatial_counter_names = ("ctr_0", "ctr_1", "ctr_2")
        """Names of the default spatial counters"""

        self.spatial_counters = (
            TypedSymbol("ctr_0", DynamicType.INDEX_TYPE),
            TypedSymbol("ctr_1", DynamicType.INDEX_TYPE),
            TypedSymbol("ctr_2", DynamicType.INDEX_TYPE),
        )
        """Default spatial counters"""

        self.index_struct_coordinate_names = ("x", "y", "z")
        """Default names of spatial coordinate members in index list structures"""

        self.sparse_counter_name = "sparse_idx"
        """Name of the default sparse iteration counter"""

        self.sparse_counter = TypedSymbol(
            self.sparse_counter_name, DynamicType.INDEX_TYPE
        )
        """Default sparse iteration counter."""

    def field_shape_name(self, field_name: str, coord: int):
        return f"_size_{field_name}_{coord}"

    def field_stride_name(self, field_name: str, coord: int):
        return f"_stride_{field_name}_{coord}"

    def field_pointer_name(self, field_name: str):
        return f"_data_{field_name}"

    def index_struct(self, index_dtype: UserTypeSpec, dim: int) -> PsStructType:
        idx_type = create_type(index_dtype)
        return PsStructType(
            [(name, idx_type) for name in self.index_struct_coordinate_names[:dim]]
        )


DEFAULTS = SympyDefaults()
"""Default names and symbols used throughout code generation"""
