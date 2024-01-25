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
    #   1. Prepare context
    ctx = KernelCreationContext(options)

    #   2. Check kernel constraints and collect knowledge
    analysis = KernelAnalysis(ctx)
    analysis(assignments)

    #   3. Create iteration space
    ispace: IterationSpace = (
        create_sparse_iteration_space(ctx, assignments)
        if len(ctx.fields.index_fields) > 0
        else create_full_iteration_space(ctx, assignments)
    )

    ctx.set_iteration_space(ispace)

    #   4. Freeze assignments
    #   This call is the same for both domain and indexed kernels
    freeze = FreezeExpressions(ctx)
    kernel_body: PsBlock = freeze(assignments)

    #   5. Typify
    #   Also the same for both types of kernels
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
