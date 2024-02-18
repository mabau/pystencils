from .backend.ast import PsKernelFunction
from .backend.kernelcreation import KernelCreationContext, KernelAnalysis, FreezeExpressions, Typifier
from .backend.kernelcreation.iteration_space import (
    create_sparse_iteration_space,
    create_full_iteration_space,
)
from .backend.kernelcreation.transformations import EraseAnonymousStructTypes

from .enums import Target
from .config import CreateKernelConfig
from pystencils.sympyextensions.assignmentcollection import AssignmentCollection


def create_kernel(
    assignments: AssignmentCollection,
    config: CreateKernelConfig = CreateKernelConfig(),
):
    """Create a kernel AST from an assignment collection."""
    ctx = KernelCreationContext(config)

    analysis = KernelAnalysis(ctx)
    analysis(assignments)

    if len(ctx.fields.index_fields) > 0 or ctx.options.index_field is not None:
        ispace = create_sparse_iteration_space(ctx, assignments)
    else:
        ispace = create_full_iteration_space(ctx, assignments)

    ctx.set_iteration_space(ispace)

    freeze = FreezeExpressions(ctx)
    kernel_body = freeze(assignments)

    typify = Typifier(ctx)
    kernel_body = typify(kernel_body)

    match config.target:
        case Target.CPU:
            from .backend.platforms import BasicCpu

            #   TODO: CPU platform should incorporate instruction set info, OpenMP, etc.
            platform = BasicCpu(ctx)
        case _:
            #   TODO: CUDA/HIP platform
            #   TODO: SYCL platform (?)
            raise NotImplementedError("Target platform not implemented")

    kernel_ast = platform.materialize_iteration_space(kernel_body, ispace)
    kernel_ast = EraseAnonymousStructTypes(ctx)(kernel_ast)

    #   7. Apply optimizations
    #     - Vectorization
    #     - OpenMP
    #     - Loop Splitting, Tiling, Blocking
    kernel_ast = platform.optimize(kernel_ast)

    assert config.jit is not None
    function = PsKernelFunction(kernel_ast, config.target, name=config.function_name, jit=config.jit)
    function.add_constraints(*ctx.constraints)

    return function
