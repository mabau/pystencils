from typing import cast, Sequence
from dataclasses import replace

from .target import Target
from .config import CreateKernelConfig
from .backend import KernelFunction
from .types import create_numeric_type, PsIntegerType
from .backend.ast.structural import PsBlock
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


from .backend.transformations import (
    EliminateConstants,
    LowerToC,
    SelectFunctions,
    CanonicalizeSymbols,
)
from .backend.kernelfunction import (
    create_cpu_kernel_function,
    create_gpu_kernel_function,
)

from .simp import AssignmentCollection
from sympy.codegen.ast import AssignmentBase


__all__ = ["create_kernel"]


def create_kernel(
    assignments: AssignmentCollection | Sequence[AssignmentBase] | AssignmentBase,
    config: CreateKernelConfig | None = None,
    **kwargs,
) -> KernelFunction:
    """Create a kernel function from a set of assignments.

    Args:
        assignments: The kernel's sequence of assignments, expressed using SymPy
        config: The configuration for the kernel translator
        kwargs: If ``config`` is not set, it is created from the keyword arguments;
            if it is set, its option will be overridden by any keyword arguments.

    Returns:
        The numerical kernel in pystencil's internal representation, ready to be
        exported or compiled
    """

    if not config:
        config = CreateKernelConfig()

    if kwargs:
        config = replace(config, **kwargs)

    idx_dtype = create_numeric_type(config.index_dtype)
    assert isinstance(idx_dtype, PsIntegerType)

    ctx = KernelCreationContext(
        default_dtype=create_numeric_type(config.default_dtype),
        index_dtype=idx_dtype,
    )

    if isinstance(assignments, AssignmentBase):
        assignments = [assignments]

    if not isinstance(assignments, AssignmentCollection):
        assignments = AssignmentCollection(assignments)  # type: ignore

    _ = _parse_simplification_hints(assignments)

    analysis = KernelAnalysis(
        ctx, not config.skip_independence_check, not config.allow_double_writes
    )
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
        case Target.GenericCPU:
            from .backend.platforms import GenericCpu

            platform = GenericCpu(ctx)
            kernel_ast = platform.materialize_iteration_space(kernel_body, ispace)

        case target if target.is_gpu():
            match target:
                case Target.SYCL:
                    from .backend.platforms import SyclPlatform

                    platform = SyclPlatform(ctx, config.gpu_indexing)
                case Target.CUDA:
                    from .backend.platforms import CudaPlatform

                    platform = CudaPlatform(ctx, config.gpu_indexing)
                case _:
                    raise NotImplementedError(
                        f"Code generation for target {target} not implemented"
                    )

            kernel_ast, gpu_threads = platform.materialize_iteration_space(
                kernel_body, ispace
            )

        case _:
            raise NotImplementedError(
                f"Code generation for target {target} not implemented"
            )

    #   Fold and extract constants
    elim_constants = EliminateConstants(ctx, extract_constant_exprs=True)
    kernel_ast = cast(PsBlock, elim_constants(kernel_ast))

    #   Target-Specific optimizations
    if config.target.is_cpu():
        from .backend.kernelcreation import optimize_cpu

        assert isinstance(platform, GenericCpu)

        kernel_ast = optimize_cpu(ctx, platform, kernel_ast, config.cpu_optim)

    #   Lowering
    lower_to_c = LowerToC(ctx)
    kernel_ast = cast(PsBlock, lower_to_c(kernel_ast))

    select_functions = SelectFunctions(platform)
    kernel_ast = cast(PsBlock, select_functions(kernel_ast))

    #   Late canonicalization and constant elimination passes
    #    * Since lowering introduces new index calculations and indexing symbols into the AST,
    #    * these need to be handled here
    
    canonicalize = CanonicalizeSymbols(ctx, True)
    kernel_ast = cast(PsBlock, canonicalize(kernel_ast))

    late_fold_constants = EliminateConstants(ctx, extract_constant_exprs=False)
    kernel_ast = cast(PsBlock, late_fold_constants(kernel_ast))

    if config.target.is_cpu():
        return create_cpu_kernel_function(
            ctx,
            platform,
            kernel_ast,
            config.function_name,
            config.target,
            config.get_jit(),
        )
    else:
        return create_gpu_kernel_function(
            ctx,
            platform,
            kernel_ast,
            gpu_threads,
            config.function_name,
            config.target,
            config.get_jit(),
        )


def create_staggered_kernel(
    assignments, target: Target = Target.CPU, gpu_exclusive_conditions=False, **kwargs
):
    raise NotImplementedError(
        "Staggered kernels are not yet implemented for pystencils 2.0"
    )


#   Internals

def _parse_simplification_hints(ac: AssignmentCollection):
    if "split_groups" in ac.simplification_hints:
        raise NotImplementedError("Loop splitting was requested, but is not implemented yet")
