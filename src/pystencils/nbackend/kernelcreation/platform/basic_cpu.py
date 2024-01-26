from pystencils.nbackend.ast import PsBlock, PsLoop, PsSymbolExpr, PsExpression
from pystencils.nbackend.kernelcreation.iteration_space import (
    IterationSpace,
    FullIterationSpace,
)
from .platform import Platform


class BasicCpu(Platform):
    def apply_iteration_space(self, block: PsBlock, ispace: IterationSpace) -> PsBlock:
        if isinstance(ispace, FullIterationSpace):
            return self._create_domain_loops(block, ispace)
        else:
            raise NotImplementedError("Iteration space not supported yet.")

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
