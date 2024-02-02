import pytest

from pystencils.field import Field, FieldType
from pystencils.nbackend.types.quick import *
from pystencils.nbackend.kernelcreation.config import (
    CreateKernelConfig,
    PsOptionsError,
)


def test_invalid_iteration_region_options():
    idx_field = Field.create_generic(
        "idx", spatial_dimensions=1, field_type=FieldType.INDEXED
    )
    with pytest.raises(PsOptionsError):
        CreateKernelConfig(
            ghost_layers=2, iteration_slice=(slice(1, -1), slice(1, -1))
        )
    with pytest.raises(PsOptionsError):
        CreateKernelConfig(ghost_layers=2, index_field=idx_field)


def test_index_field_options():
    with pytest.raises(PsOptionsError):
        idx_field = Field.create_generic(
            "idx", spatial_dimensions=1, field_type=FieldType.GENERIC
        )
        CreateKernelConfig(index_field=idx_field)
