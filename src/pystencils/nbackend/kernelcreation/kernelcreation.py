from ...simp import AssignmentCollection

from ..ast import PsBlock, PsKernelFunction
from ...enums import Target

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
    match options.target:
        case Target.CPU:
            from .platform import BasicCpu

            #   TODO: CPU platform should incorporate instruction set info, OpenMP, etc.
            platform = BasicCpu(ctx)
        case _:
            #   TODO: CUDA/HIP platform
            #   TODO: SYCL platform (?)
            raise NotImplementedError("Target platform not implemented")

    #   6. Add loops or device indexing
    kernel_ast = platform.apply_iteration_space(kernel_body, ispace)

    #   7. Apply optimizations
    #     - Vectorization
    #     - OpenMP
    #     - Loop Splitting, Tiling, Blocking
    kernel_ast = platform.optimize(kernel_ast)

    #   8. Create and return kernel function.
    function = PsKernelFunction(kernel_ast, options.target, name=options.function_name)
    function.add_constraints(*ctx.constraints)

    return function
