from ...simp import AssignmentCollection

from ..ast import PsKernelFunction
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
        if len(ctx.fields.index_fields) > 0 or ctx.options.index_field is not None
        else create_full_iteration_space(ctx, assignments)
    )

    ctx.set_iteration_space(ispace)

    freeze = FreezeExpressions(ctx)
    kernel_body = freeze(assignments)

    typify = Typifier(ctx)
    kernel_body = typify(kernel_body)

    match options.target:
        case Target.CPU:
            from .platform import BasicCpuGen

            #   TODO: CPU platform should incorporate instruction set info, OpenMP, etc.
            platform_generator = BasicCpuGen(ctx)
        case _:
            #   TODO: CUDA/HIP platform
            #   TODO: SYCL platform (?)
            raise NotImplementedError("Target platform not implemented")

    kernel_ast = platform_generator.materialize_iteration_space(kernel_body, ispace)

    #   7. Apply optimizations
    #     - Vectorization
    #     - OpenMP
    #     - Loop Splitting, Tiling, Blocking
    kernel_ast = platform_generator.optimize(kernel_ast)

    function = PsKernelFunction(kernel_ast, options.target, name=options.function_name)
    function.add_constraints(*ctx.constraints)

    return function