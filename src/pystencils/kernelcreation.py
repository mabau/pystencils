from typing import Sequence
from dataclasses import dataclass

from .enums import Target
from .field import Field, FieldType

from .backend.jit import JitBase
from .backend.exceptions import PsOptionsError
from .backend.types import PsIntegerType, PsNumericType, PsIeeeFloatType

from .backend.kernelcreation.defaults import Sympy as SpDefaults

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
from .backend.kernelcreation.transformations import EraseAnonymousStructTypes

from .enums import Target
from .sympyextensions import AssignmentCollection


@dataclass
class CreateKernelConfig:
    """Options for create_kernel."""

    target: Target = Target.CPU
    """The code generation target.
    
    TODO: Enhance `Target` from enum to a larger target spec, e.g. including vectorization architecture, ...
    """

    jit: JitBase | None = None
    """Just-in-time compiler used to compile and load the kernel for invocation from the current Python environment.
    
    If left at `None`, a default just-in-time compiler will be inferred from the `target` parameter.
    To explicitly disable JIT compilation, pass `nbackend.jit.no_jit`.
    """

    function_name: str = "kernel"
    """Name of the generated function"""

    ghost_layers: None | int | Sequence[int | tuple[int, int]] = None
    """Specifies the number of ghost layers of the iteration region.
    
    Options:
     - `None`: Required ghost layers are inferred from field accesses
     - `int`:  A uniform number of ghost layers in each spatial coordinate is applied
     - `Sequence[int, tuple[int, int]]`: Ghost layers are specified for each spatial coordinate.
        In each coordinate, a single integer specifies the ghost layers at both the lower and upper iteration limit,
        while a pair of integers specifies the lower and upper ghost layers separately.

    When manually specifying ghost layers, it is the user's responsibility to avoid out-of-bounds memory accesses.
    If `ghost_layers=None` is specified, the iteration region may otherwise be set using the `iteration_slice` option.
    """

    iteration_slice: None | tuple[slice, ...] = None
    """Specifies the kernel's iteration slice.
    
    `iteration_slice` may only be set if `ghost_layers = None`.
    If it is set, a slice must be specified for each spatial coordinate.
    TODO: Specification of valid slices and their behaviour
    """

    index_field: Field | None = None
    """Index field for a sparse kernel.
    
    If this option is set, a sparse kernel with the given field as index field will be generated.
    """

    """Data Types"""

    index_dtype: PsIntegerType = SpDefaults.index_dtype
    """Data type used for all index calculations."""

    default_dtype: PsNumericType = PsIeeeFloatType(64)
    """Default numeric data type.
    
    This data type will be applied to all untyped symbols.
    """

    def __post_init__(self):
        #   Check iteration space argument consistency
        if (
            int(self.iteration_slice is not None)
            + int(self.ghost_layers is not None)
            + int(self.index_field is not None)
            > 1
        ):
            raise PsOptionsError(
                "Parameters `iteration_slice`, `ghost_layers` and 'index_field` are mutually exclusive; "
                "at most one of them may be set."
            )

        #   Check index field
        if (
            self.index_field is not None
            and self.index_field.field_type != FieldType.INDEXED
        ):
            raise PsOptionsError(
                "Only fields with `field_type == FieldType.INDEXED` can be specified as `index_field`"
            )

        #   Infer JIT
        if self.jit is None:
            match self.target:
                case Target.CPU:
                    from .backend.jit import LegacyCpuJit

                    self.jit = LegacyCpuJit()
                case _:
                    raise NotImplementedError(
                        f"No default JIT compiler implemented yet for target {self.target}"
                    )


def create_kernel(
    assignments: AssignmentCollection,
    config: CreateKernelConfig = CreateKernelConfig(),
):
    """Create a kernel AST from an assignment collection."""
    ctx = KernelCreationContext(
        default_dtype=config.default_dtype, index_dtype=config.index_dtype
    )

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
    function = PsKernelFunction(
        kernel_ast, config.target, name=config.function_name, jit=config.jit
    )
    function.add_constraints(*ctx.constraints)

    return function
