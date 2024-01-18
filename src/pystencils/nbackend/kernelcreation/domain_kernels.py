from types import EllipsisType

from ...simp import AssignmentCollection
from ...field import Field
from ...kernel_contrains_check import KernelConstraintsCheck

from ..types.quick import SInt
from ..ast import PsBlock

from .context import KernelCreationContext, FullIterationSpace
from .freeze import FreezeExpressions

# flake8: noqa
def create_domain_kernel(assignments: AssignmentCollection):
    #   TODO: Assemble configuration

    #   1. Prepare context
    ctx = KernelCreationContext(SInt(64))  # TODO: how to determine index type?

    #   2. Check kernel constraints and collect all fields
    check = KernelConstraintsCheck()  # TODO: config
    check.visit(assignments)

    #   All steps up to this point are the same in domain and indexed kernels;
    #   the difference now comes with the iteration space.
    #
    #   Domain kernels create a full iteration space from their iteration slice
    #   which is either explicitly given or computed from ghost layer requirements.
    #   Indexed kernels, on the other hand, have to create a sparse iteration space
    #   from one index list.

    #   3. Create iteration space
    ghost_layers: int = NotImplemented  # determine required ghost layers
    common_shape: tuple[
        int | EllipsisType, ...
    ] = NotImplemented  # unify field shapes, add parameter constraints
    #   don't forget custom iteration slice
    ispace: FullIterationSpace = (
        NotImplemented  # create from ghost layers and with given shape
    )

    #   4. Freeze assignments
    #   This call is the same for both domain and indexed kernels
    freeze = FreezeExpressions(ctx)
    kernel_body: PsBlock = freeze(assignments)

    #   5. Typify
    #   Also the same for both types of kernels
    #   determine_types(kernel_body)

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
