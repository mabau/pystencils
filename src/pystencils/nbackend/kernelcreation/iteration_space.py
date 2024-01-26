from __future__ import annotations
from typing import Sequence, TYPE_CHECKING
from abc import ABC
from dataclasses import dataclass
from functools import reduce
from operator import mul

from ...simp import AssignmentCollection
from ...field import Field, FieldType

from ..typed_expressions import (
    PsTypedVariable,
    VarOrConstant,
    ExprOrConstant,
    PsTypedConstant,
)
from ..arrays import PsLinearizedArray
from .defaults import Pymbolic as Defaults
from ..exceptions import PsInputError, PsInternalCompilerError

if TYPE_CHECKING:
    from .context import KernelCreationContext


class IterationSpace(ABC):
    """Represents the n-dimensonal iteration space of a pystencils kernel.

    Instances of this class represent the kernel's iteration region during translation from
    SymPy, before any indexing sources are generated. It provides the counter symbols which
    should be used to translate field accesses to array accesses.

    There are two types of iteration spaces, modelled by subclasses:
     - The full iteration space translates to an n-dimensional loop nest or the corresponding device
       indexing scheme.
     - The sparse iteration space translates to a single loop over an index list which in turn provides the
       spatial indices.
    """

    def __init__(self, spatial_indices: tuple[PsTypedVariable, ...]):
        if len(spatial_indices) == 0:
            raise ValueError("Iteration space must be at least one-dimensional.")

        self._spatial_indices = spatial_indices

    @property
    def spatial_indices(self) -> tuple[PsTypedVariable, ...]:
        return self._spatial_indices


class FullIterationSpace(IterationSpace):
    """N-dimensional full iteration space.
    
    Each dimension of the full iteration space is represented by an instance of `FullIterationSpace.Dimension`.
    Dimensions are ordered slowest-to-fastest: The first dimension corresponds to the slowest coordinate,
    translates to the outermost loop, while the last dimension is the fastest coordinate and translates
    to the innermost loop.
    """

    @dataclass
    class Dimension:
        start: VarOrConstant
        stop: VarOrConstant
        step: VarOrConstant
        counter: PsTypedVariable

    @staticmethod
    def create_with_ghost_layers(
        ctx: KernelCreationContext,
        archetype_field: Field,
        ghost_layers: int | Sequence[int | tuple[int, int]],
    ) -> FullIterationSpace:
        """Create an iteration space for a collection of fields with ghost layers."""

        archetype_array = ctx.get_array(archetype_field)
        dim = archetype_field.spatial_dimensions
        counters = [
            PsTypedVariable(name, ctx.index_dtype)
            for name in Defaults.spatial_counter_names[:dim]
        ]

        if isinstance(ghost_layers, int):
            ghost_layers_spec = [(ghost_layers, ghost_layers) for _ in range(dim)]
        else:
            if len(ghost_layers) != dim:
                raise ValueError("Too few entries in ghost layer spec")
            ghost_layers_spec = [
                ((gl, gl) if isinstance(gl, int) else gl) for gl in ghost_layers
            ]

        one = PsTypedConstant(1, ctx.index_dtype)

        ghost_layer_exprs = [
            (
                PsTypedConstant(gl_left, ctx.index_dtype),
                PsTypedConstant(gl_right, ctx.index_dtype),
            )
            for (gl_left, gl_right) in ghost_layers_spec
        ]

        spatial_shape = archetype_array.shape[:dim]

        dimensions = [
            FullIterationSpace.Dimension(gl_left, shape - gl_right, one, ctr)
            for (gl_left, gl_right), shape, ctr in zip(
                ghost_layer_exprs, spatial_shape, counters, strict=True
            )
        ]

        #   TODO: Reorder dimensions according to optimal loop layout (?)

        return FullIterationSpace(ctx, dimensions)

    def __init__(self, ctx: KernelCreationContext, dimensions: Sequence[Dimension]):
        super().__init__(tuple(dim.counter for dim in dimensions))

        self._ctx = ctx
        self._dimensions = dimensions

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def lower(self):
        return (dim.start for dim in self._dimensions)

    @property
    def upper(self):
        return (dim.stop for dim in self._dimensions)

    @property
    def steps(self):
        return (dim.step for dim in self._dimensions)

    def num_iteration_items(self, dimension: int | None = None) -> ExprOrConstant:
        if dimension is None:
            return reduce(
                mul, (self.num_iteration_items(d) for d in range(len(self.dimensions)))
            )
        else:
            dim = self.dimensions[dimension]
            one = PsTypedConstant(1, self._ctx.index_dtype)
            return one + (dim.stop - dim.start - one) / dim.step


class SparseIterationSpace(IterationSpace):
    #   TODO: To properly implement sparse iteration, we still need struct data types
    def __init__(
        self,
        spatial_indices: tuple[PsTypedVariable, ...],
        index_list: PsLinearizedArray,
    ):
        super().__init__(spatial_indices)
        self._index_list = index_list

    @property
    def index_list(self) -> PsLinearizedArray:
        return self._index_list


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

    #   TODO: Re-implement as `get_archetype_field`, check not only shape but also layout equality
    #   The archetype field must encompass all information about the iteration space: shape, extents, and loop order.
    from ...transformations import get_common_field

    if len(domain_field_accesses) > 0:
        archetype_field = get_common_field(ctx.fields.domain_fields)
        inferred_gls = max([fa.required_ghost_layers for fa in domain_field_accesses])
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
