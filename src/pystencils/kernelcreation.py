from .enums import Target
from .config import CreateKernelConfig
from .backend.ast import PsKernelFunction
from .backend.kernelcreation import (
    KernelCreationContext,
    KernelAnalysis,
    FreezeExpressions,
    Typifier,
)
from .backend.kernelcreation.iteration_space import (
    create_sparse_iteration_space,
    create_full_iteration_space,
)

from .backend.ast.analysis import collect_required_headers
from .backend.transformations import EraseAnonymousStructTypes

from .enums import Target
from .sympyextensions import AssignmentCollection, Assignment


def create_kernel(
    assignments: AssignmentCollection | list[Assignment],
    config: CreateKernelConfig = CreateKernelConfig(),
):
    """Create a kernel AST from an assignment collection."""
    ctx = KernelCreationContext(
        default_dtype=config.default_dtype, index_dtype=config.index_dtype
    )

    if not isinstance(assignments, AssignmentCollection):
        assignments = AssignmentCollection(assignments)

    analysis = KernelAnalysis(ctx)
    analysis(assignments)

    if len(ctx.fields.index_fields) > 0 or config.index_field is not None:
        ispace = create_sparse_iteration_space(
            ctx, assignments, index_field=config.index_field
        )
    else:
        ispace = create_full_iteration_space(
            ctx,
            assignments,
            ghost_layers=config.ghost_layers,
            iteration_slice=config.iteration_slice,
        )

    ctx.set_iteration_space(ispace)

    freeze = FreezeExpressions(ctx)
    kernel_body = freeze(assignments)

    typify = Typifier(ctx)
    kernel_body = typify(kernel_body)

    match config.target:
        case Target.CPU:
            from .backend.platforms import GenericCpu

            #   TODO: CPU platform should incorporate instruction set info, OpenMP, etc.
            platform = GenericCpu(ctx)
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
    req_headers = collect_required_headers(kernel_ast) | platform.required_headers
    function = PsKernelFunction(
        kernel_ast, config.target, config.function_name, req_headers, jit=config.jit
    )
    function.add_constraints(*ctx.constraints)

    return function
