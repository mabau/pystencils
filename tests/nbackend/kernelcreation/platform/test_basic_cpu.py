import pytest

from pystencils.field import Field

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    FullIterationSpace
)

from pystencils.backend.ast import PsBlock, PsLoop, PsComment, dfs_preorder

from pystencils.backend.platforms import BasicCpu

@pytest.mark.parametrize("layout", ["fzyx", "zyxf", "c", "f"])
def test_loop_nest(layout):
    ctx = KernelCreationContext()

    body = PsBlock([PsComment("Loop body goes here")])
    platform = BasicCpu(ctx)

    #   FZYX Order
    archetype_field = Field.create_generic("fzyx_field", spatial_dimensions=3, layout=layout)
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, archetype_field, 0)

    loop_nest = platform.materialize_iteration_space(body, ispace)

    loops = dfs_preorder(loop_nest, lambda n: isinstance(n, PsLoop))
    for loop, dim in zip(loops, ispace.dimensions, strict=True):
        assert isinstance(loop, PsLoop)
        assert loop.start.expression == dim.start
        assert loop.stop.expression == dim.stop
        assert loop.step.expression == dim.step
        assert loop.counter.expression == dim.counter
