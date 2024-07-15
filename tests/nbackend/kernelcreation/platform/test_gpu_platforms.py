import pytest

from pystencils.field import Field

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    FullIterationSpace
)

from pystencils.backend.ast.structural import PsBlock, PsComment

from pystencils.backend.platforms import CudaPlatform, SyclPlatform


@pytest.mark.parametrize("layout", ["fzyx", "zyxf", "c", "f"])
@pytest.mark.parametrize("platform_class", [CudaPlatform, SyclPlatform])
def test_thread_range(platform_class, layout):
    ctx = KernelCreationContext()

    body = PsBlock([PsComment("Kernel body goes here")])
    platform = platform_class(ctx)

    dim = 3
    archetype_field = Field.create_generic("field", spatial_dimensions=dim, layout=layout)
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, 1, archetype_field)

    _, threads_range = platform.materialize_iteration_space(body, ispace)

    assert threads_range.dim == dim
    
    match layout:
        case "fzyx" | "zyxf" | "f":
            indexing_order = [0, 1, 2]
        case "c":
            indexing_order = [2, 1, 0]

    for i in range(dim):
        #   Slowest to fastest coordinate
        coordinate = indexing_order[i]
        dimension = ispace.dimensions[coordinate]
        witems = threads_range.num_work_items[i]
        desired = dimension.stop - dimension.start
        assert witems.structurally_equal(desired)
