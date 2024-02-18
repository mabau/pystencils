from .platform import Platform

from ..kernelcreation.iteration_space import (
    IterationSpace,
    FullIterationSpace,
    SparseIterationSpace,
)

from ..ast import PsDeclaration, PsSymbolExpr, PsExpression, PsLoop, PsBlock
from ..typed_expressions import PsTypedConstant
from ..arrays import PsArrayAccess


class BasicCpu(Platform):
    def materialize_iteration_space(
        self, body: PsBlock, ispace: IterationSpace
    ) -> PsBlock:
        if isinstance(ispace, FullIterationSpace):
            return self._create_domain_loops(body, ispace)
        elif isinstance(ispace, SparseIterationSpace):
            return self._create_sparse_loop(body, ispace)
        else:
            assert False, "unreachable code"

    def optimize(self, kernel: PsBlock) -> PsBlock:
        return kernel

    #   Internals

    def _create_domain_loops(
        self, body: PsBlock, ispace: FullIterationSpace
    ) -> PsBlock:
        dimensions = ispace.dimensions
        outer_block = body

        for dimension in dimensions[::-1]:
            loop = PsLoop(
                PsSymbolExpr(dimension.counter),
                PsExpression(dimension.start),
                PsExpression(dimension.stop),
                PsExpression(dimension.step),
                outer_block,
            )
            outer_block = PsBlock([loop])

        return outer_block

    def _create_sparse_loop(self, body: PsBlock, ispace: SparseIterationSpace):
        mappings = [
            PsDeclaration(
                PsSymbolExpr(ctr),
                PsExpression(
                    PsArrayAccess(
                        ispace.index_list.base_pointer, ispace.sparse_counter
                    ).a.__getattr__(coord.name)
                ),
            )
            for ctr, coord in zip(ispace.spatial_indices, ispace.coordinate_members)
        ]

        body = PsBlock(mappings + body.statements)

        loop = PsLoop(
            PsSymbolExpr(ispace.sparse_counter),
            PsExpression(PsTypedConstant(0, self._ctx.index_dtype)),
            PsExpression(ispace.index_list.shape[0]),
            PsExpression(PsTypedConstant(1, self._ctx.index_dtype)),
            body,
        )

        return PsBlock([loop])
