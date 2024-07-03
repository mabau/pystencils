from typing import cast

from .enums import Target
from .config import CreateKernelConfig
from .types import create_numeric_type
from .backend import (
    KernelFunction,
    KernelParameter,
    FieldShapeParam,
    FieldStrideParam,
    FieldPointerParam,
)
from .backend.symbols import PsSymbol
from .backend.jit import JitBase
from .backend.ast.structural import PsBlock
from .backend.arrays import PsArrayShapeSymbol, PsArrayStrideSymbol, PsArrayBasePointer
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

from .backend.ast.analysis import collect_required_headers, collect_undefined_symbols
from .backend.transformations import (
    EliminateConstants,
    EraseAnonymousStructTypes,
    SelectFunctions,
)

from .sympyextensions import AssignmentCollection, Assignment


__all__ = ["create_kernel"]


def create_kernel(
    assignments: AssignmentCollection | list[Assignment] | Assignment,
    config: CreateKernelConfig = CreateKernelConfig(),
) -> KernelFunction:
    """Create a kernel function from a set of assignments.

    Args:
        assignments: The kernel's sequence of assignments, expressed using SymPy
        config: The configuration for the kernel translator

    Returns:
        The numerical kernel in pystencil's internal representation, ready to be
        exported or compiled
    """

    ctx = KernelCreationContext(
        default_dtype=create_numeric_type(config.default_dtype), index_dtype=config.index_dtype
    )

    if isinstance(assignments, Assignment):
        assignments = [assignments]

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
        case Target.GenericCPU:
            from .backend.platforms import GenericCpu

            platform = GenericCpu(ctx)
        case _:
            #   TODO: CUDA/HIP platform
            #   TODO: SYCL platform (?)
            raise NotImplementedError("Target platform not implemented")

    kernel_ast = platform.materialize_iteration_space(kernel_body, ispace)

    #   Simplifying transformations
    elim_constants = EliminateConstants(ctx, extract_constant_exprs=True)
    kernel_ast = cast(PsBlock, elim_constants(kernel_ast))

    #   Target-Specific optimizations
    if config.target.is_cpu():
        from .backend.kernelcreation import optimize_cpu

        kernel_ast = optimize_cpu(ctx, platform, kernel_ast, config.cpu_optim)

    erase_anons = EraseAnonymousStructTypes(ctx)
    kernel_ast = cast(PsBlock, erase_anons(kernel_ast))

    select_functions = SelectFunctions(platform)
    kernel_ast = cast(PsBlock, select_functions(kernel_ast))

    assert config.jit is not None
    return create_kernel_function(
        ctx, kernel_ast, config.function_name, config.target, config.jit
    )


def create_kernel_function(
    ctx: KernelCreationContext,
    body: PsBlock,
    function_name: str,
    target_spec: Target,
    jit: JitBase,
):
    undef_symbols = collect_undefined_symbols(body)

    params = []
    for symb in undef_symbols:
        match symb:
            case PsArrayShapeSymbol(name, _, arr, coord):
                field = ctx.find_field(arr.name)
                params.append(FieldShapeParam(name, symb.get_dtype(), field, coord))
            case PsArrayStrideSymbol(name, _, arr, coord):
                field = ctx.find_field(arr.name)
                params.append(FieldStrideParam(name, symb.get_dtype(), field, coord))
            case PsArrayBasePointer(name, _, arr):
                field = ctx.find_field(arr.name)
                params.append(FieldPointerParam(name, symb.get_dtype(), field))
            case PsSymbol(name, _):
                params.append(KernelParameter(name, symb.get_dtype()))

    params.sort(key=lambda p: p.name)

    req_headers = collect_required_headers(body)
    req_headers |= ctx.required_headers

    return KernelFunction(
        body, target_spec, function_name, params, req_headers, ctx.constraints, jit
    )
