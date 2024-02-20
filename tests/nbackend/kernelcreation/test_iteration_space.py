import pytest

import pymbolic.primitives as pb
from pystencils.field import Field
from pystencils.sympyextensions.typed_sympy import TypedSymbol, create_type

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    FullIterationSpace
)

from pystencils.backend.kernelcreation.typification import TypificationError
from pystencils.backend.kernelcreation.defaults import Pymbolic as PbDefaults

from pystencils.backend.typed_expressions import PsTypedConstant


def test_loop_order():
    ctx = KernelCreationContext()
    ctr_symbols = PbDefaults.spatial_counters

    #   FZYX Order
    archetype_field = Field.create_generic("fzyx_field", spatial_dimensions=3, layout='fzyx')
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, archetype_field, 0)

    for dim, ctr in zip(ispace.dimensions, ctr_symbols[::-1]):
        assert dim.counter == ctr

    #   ZYXF Order
    archetype_field = Field.create_generic("zyxf_field", spatial_dimensions=3, layout='zyxf')
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, archetype_field, 0)

    for dim, ctr in zip(ispace.dimensions, ctr_symbols[::-1]):
        assert dim.counter == ctr

    #   C Order
    archetype_field = Field.create_generic("c_field", spatial_dimensions=3, layout='c')
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, archetype_field, 0)

    for dim, ctr in zip(ispace.dimensions, ctr_symbols):
        assert dim.counter == ctr

    #   Fortran Order
    archetype_field = Field.create_generic("fortran_field", spatial_dimensions=3, layout='f')
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, archetype_field, 0)

    for dim, ctr in zip(ispace.dimensions, ctr_symbols[::-1]):
        assert dim.counter == ctr

    #   Scrambled Layout
    archetype_field = Field.create_generic("scrambled_field", spatial_dimensions=3, layout=(2, 0, 1))
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, archetype_field, 0)

    for dim, ctr in zip(ispace.dimensions, [ctr_symbols[2], ctr_symbols[0], ctr_symbols[1]]):
        assert dim.counter == ctr


def test_slices():
    ctx = KernelCreationContext()
    ctr_symbols = PbDefaults.spatial_counters

    archetype_field = Field.create_generic("f", spatial_dimensions=3, layout='fzyx')
    ctx.add_field(archetype_field)

    islice = (slice(1, -1, 1), slice(3, -3, 3), slice(0, None, -1))
    ispace = FullIterationSpace.create_from_slice(ctx, archetype_field, islice)

    archetype_arr = ctx.get_array(archetype_field)

    dims = ispace.dimensions[::-1]

    for sl, size, dim in zip(islice, archetype_arr.shape, dims):
        assert isinstance(dim.start, PsTypedConstant) and dim.start.value == sl.start
        assert isinstance(dim.step, PsTypedConstant) and dim.step.value == sl.step

    assert isinstance(dims[0].stop, pb.Sum) and archetype_arr.shape[0] in dims[0].stop.children
    assert isinstance(dims[1].stop, pb.Sum) and archetype_arr.shape[1] in dims[1].stop.children
    assert dims[2].stop == archetype_arr.shape[2]


def test_invalid_slices():
    ctx = KernelCreationContext()
    ctr_symbols = PbDefaults.spatial_counters

    archetype_field = Field.create_generic("f", spatial_dimensions=1, layout='fzyx')
    ctx.add_field(archetype_field)

    islice = (slice(1, -1, 0.5),)
    with pytest.raises(ValueError):
        FullIterationSpace.create_from_slice(ctx, archetype_field, islice)

    islice = (slice(1, -1, TypedSymbol("w", dtype=create_type("double"))),)
    with pytest.raises(TypificationError):
        FullIterationSpace.create_from_slice(ctx, archetype_field, islice)
