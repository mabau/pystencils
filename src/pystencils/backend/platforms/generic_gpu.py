from .platform import Platform

from ..kernelcreation.iteration_space import (
    IterationSpace,
    FullIterationSpace,
    # SparseIterationSpace,
)

from ..ast.structural import PsBlock, PsConditional
from ..ast.expressions import (
    PsExpression,
    PsSymbolExpr,
    PsAdd,
)
from ..ast.logical_expressions import PsLt, PsAnd
from ...types import PsSignedIntegerType
from ..symbols import PsSymbol

int32 = PsSignedIntegerType(width=32, const=False)

BLOCK_IDX = [
    PsSymbolExpr(PsSymbol(f"blockIdx.{coord}", int32)) for coord in ("x", "y", "z")
]
THREAD_IDX = [
    PsSymbolExpr(PsSymbol(f"threadIdx.{coord}", int32)) for coord in ("x", "y", "z")
]
BLOCK_DIM = [
    PsSymbolExpr(PsSymbol(f"blockDim.{coord}", int32)) for coord in ("x", "y", "z")
]
GRID_DIM = [
    PsSymbolExpr(PsSymbol(f"gridDim.{coord}", int32)) for coord in ("x", "y", "z")
]


class GenericGpu(Platform):

    @property
    def required_headers(self) -> set[str]:
        return {"gpu_defines.h"}

    def materialize_iteration_space(
        self, body: PsBlock, ispace: IterationSpace
    ) -> PsBlock:
        if isinstance(ispace, FullIterationSpace):
            return self._guard_full_iteration_space(body, ispace)
        else:
            assert False, "unreachable code"

    def cuda_indices(self, dim):
        block_size = BLOCK_DIM
        indices = [
            block_index * bs + thread_idx
            for block_index, bs, thread_idx in zip(BLOCK_IDX, block_size, THREAD_IDX)
        ]

        return indices[:dim]

    #   Internals
    def _guard_full_iteration_space(
        self, body: PsBlock, ispace: FullIterationSpace
    ) -> PsBlock:

        dimensions = ispace.dimensions

        #   Determine loop order by permuting dimensions
        archetype_field = ispace.archetype_field
        if archetype_field is not None:
            loop_order = archetype_field.layout
            dimensions = [dimensions[coordinate] for coordinate in loop_order]

        start = [
            PsAdd(c, d.start)
            for c, d in zip(self.cuda_indices(len(dimensions)), dimensions[::-1])
        ]
        conditions = [PsLt(c, d.stop) for c, d in zip(start, dimensions[::-1])]

        condition: PsExpression = conditions[0]
        for c in conditions[1:]:
            condition = PsAnd(condition, c)

        return PsBlock([PsConditional(condition, body)])
