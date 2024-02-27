import pytest

from pystencils.field import Field

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    FullIterationSpace
)

from pystencils.backend.ast.structural import PsBlock, PsLoop, PsComment
from pystencils.backend.ast.expressions import PsExpression
from pystencils.backend.ast import dfs_preorder

from pystencils.backend.platforms import GenericCpu

@pytest.mark.parametrize("layout", ["fzyx", "zyxf", "c", "f"])
def test_loop_nest(layout):
    ctx = KernelCreationContext()

    body = PsBlock([PsComment("Loop body goes here")])
    platform = GenericCpu(ctx)

    #   FZYX Order
    archetype_field = Field.create_generic("fzyx_field", spatial_dimensions=3, layout=layout)
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, archetype_field, 0)

    loop_nest = platform.materialize_iteration_space(body, ispace)

    loops = dfs_preorder(loop_nest, lambda n: isinstance(n, PsLoop))
    for loop, dim in zip(loops, ispace.dimensions, strict=True):
        assert isinstance(loop, PsLoop)
        assert loop.start.structurally_equal(dim.start)
        assert loop.stop.structurally_equal(dim.stop)
        assert loop.step.structurally_equal(dim.step)
        assert loop.counter.structurally_equal(PsExpression.make(dim.counter))
