from itertools import chain

from ...simp import AssignmentCollection

from ..ast import PsBlock

from .context import KernelCreationContext
from .analysis import KernelAnalysis
from .freeze import FreezeExpressions
from .typification import Typifier
from .options import KernelCreationOptions
from .iteration_space import (
    IterationSpace,
    create_sparse_iteration_space,
    create_full_iteration_space,
)

# flake8: noqa


def create_kernel(assignments: AssignmentCollection, options: KernelCreationOptions):
    ctx = KernelCreationContext(options)

    analysis = KernelAnalysis(ctx)
    analysis(assignments)

    ispace: IterationSpace = (
        create_sparse_iteration_space(ctx, assignments)
        if len(ctx.fields.index_fields) > 0
        else create_full_iteration_space(ctx, assignments)
    )

    ctx.set_iteration_space(ispace)

    freeze = FreezeExpressions(ctx)
    kernel_body: PsBlock = freeze(assignments)

    typify = Typifier(ctx)
    kernel_body = typify(kernel_body)

    #   Up to this point, all was target-agnostic, but now the target becomes relevant.
    #   Here we might hand off the compilation to a target-specific part of the compiler
    #   (CPU/CUDA/...), since these will likely also apply very different optimizations.

    #   6. Add loops or device indexing
    #   This step translates the iteration space to actual index calculation code and is once again
    #   different in indexed and domain kernels.

    #   7. Apply optimizations
    #     - Vectorization
    #     - OpenMP
    #     - Loop Splitting, Tiling, Blocking

    #   8. Create and return kernel function.
