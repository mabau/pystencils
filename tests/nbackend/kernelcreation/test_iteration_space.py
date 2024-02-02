from pystencils.field import Field

from pystencils.nbackend.kernelcreation import (
    KernelCreationContext,
    CreateKernelConfig,
    FullIterationSpace
)

from pystencils.nbackend.kernelcreation.defaults import Pymbolic as PbDefaults


def test_loop_order():
    ctx = KernelCreationContext(CreateKernelConfig())
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
