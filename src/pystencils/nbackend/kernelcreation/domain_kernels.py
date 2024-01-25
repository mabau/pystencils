from typing import Sequence
from itertools import chain

from ...simp import AssignmentCollection
from ...field import Field, FieldType
from ...kernel_contrains_check import KernelConstraintsCheck

from ..ast import PsBlock

from .context import KernelCreationContext, IterationSpace
from .freeze import FreezeExpressions
from .typification import Typifier
from .options import KernelCreationOptions
from ..exceptions import PsInputError, PsInternalCompilerError

# flake8: noqa


def create_kernel(assignments: AssignmentCollection, options: KernelCreationOptions):
    #   1. Prepare context
    ctx = KernelCreationContext(options)

    #   2. Check kernel constraints and collect all fields

    """
    TODO: Replace the KernelConstraintsCheck by a KernelAnalysis pass.

    The kernel analysis should:
     - Check constraints on the assignments (SSA form, independence conditions, ...)
     - Collect all fields and register them in the context
     - Maybe collect all field accesses and register them at the context
     
    Running all this analysis in a single pass will likely improve compiler performance
    since no additional searches, e.g. for field accesses, are necessary later.
    """

    check = KernelConstraintsCheck()  # TODO: config
    check.visit(assignments)

    #   Collect all fields
    for f in chain(check.fields_written, check.fields_read):
        ctx.add_field(f)

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


def create_sparse_iteration_space(
    ctx: KernelCreationContext, assignments: AssignmentCollection
) -> IterationSpace:
    return NotImplemented


def create_full_iteration_space(
    ctx: KernelCreationContext, assignments: AssignmentCollection
) -> IterationSpace:
    assert not ctx.fields.index_fields

    #   Collect all relative accesses into domain fields
    def access_filter(acc: Field.Access):
        return acc.field.field_type in (
            FieldType.GENERIC,
            FieldType.STAGGERED,
            FieldType.STAGGERED_FLUX,
        )

    domain_field_accesses = assignments.atoms(Field.Access)
    domain_field_accesses = set(filter(access_filter, domain_field_accesses))

    # The following scenarios exist:
    # - We have at least one domain field -> find the common field and use it to determine the iteration region
    # - We have no domain fields, but at least one custom field -> determine common field from custom fields
    # - We have neither domain nor custom fields -> Error

    from ...transformations import get_common_field

    if len(domain_field_accesses) > 0:
        archetype_field = get_common_field(ctx.fields.domain_fields)
        inferred_gls = max(
            [fa.required_ghost_layers for fa in domain_field_accesses]
        )
    elif len(ctx.fields.custom_fields) > 0:
        archetype_field = get_common_field(ctx.fields.custom_fields)
        inferred_gls = 0
    else:
        raise PsInputError(
            "Unable to construct iteration space: The kernel contains no accesses to domain or custom fields."
        )

    # If the user provided a ghost layer specification, use that
    # Otherwise, if an iteration slice was specified, use that
    # Otherwise, use the inferred ghost layers

    from .iteration_space import FullIterationSpace

    if ctx.options.ghost_layers is not None:
        return FullIterationSpace.create_with_ghost_layers(
            ctx, archetype_field, ctx.options.ghost_layers
        )
    elif ctx.options.iteration_slice is not None:
        raise PsInternalCompilerError("Iteration slices not supported yet")
    else:
        return FullIterationSpace.create_with_ghost_layers(
            ctx, archetype_field, inferred_gls
        )
