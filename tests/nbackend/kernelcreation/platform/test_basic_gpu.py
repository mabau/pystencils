import pytest

from pystencils.field import Field

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    FullIterationSpace
)

from pystencils.backend.ast.structural import PsBlock, PsLoop, PsComment
from pystencils.backend.ast.expressions import PsExpression
from pystencils.backend.ast import dfs_preorder

from pystencils.backend.platforms import GenericGpu


@pytest.mark.parametrize("layout", ["fzyx", "zyxf", "c", "f"])
def test_loop_nest(layout):
    ctx = KernelCreationContext()

    body = PsBlock([PsComment("Loop body goes here")])
    platform = GenericGpu(ctx)

    #   FZYX Order
    archetype_field = Field.create_generic("fzyx_field", spatial_dimensions=3, layout=layout)
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, 0, archetype_field)

    condition = platform.materialize_iteration_space(body, ispace)
