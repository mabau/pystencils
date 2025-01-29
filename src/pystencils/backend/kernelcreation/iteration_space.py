from __future__ import annotations
from typing import Sequence, TYPE_CHECKING
from abc import ABC
from dataclasses import dataclass
from functools import reduce
from operator import mul

from ...defaults import DEFAULTS
from ...simp import AssignmentCollection
from ...field import Field, FieldType

from ..memory import PsSymbol, PsBuffer
from ..constants import PsConstant
from ..ast.expressions import PsExpression, PsConstantExpr, PsTernary, PsEq, PsRem
from ..ast.util import failing_cast
from ...types import PsStructType
from ..exceptions import PsInputError, KernelConstraintsError

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

    def __init__(self, spatial_indices: Sequence[PsSymbol]):
        if len(spatial_indices) == 0:
            raise ValueError("Iteration space must be at least one-dimensional.")

        self._spatial_indices = tuple(spatial_indices)

    @property
    def spatial_indices(self) -> tuple[PsSymbol, ...]:
        return self._spatial_indices

    @property
    def rank(self) -> int:
        return len(self._spatial_indices)


class FullIterationSpace(IterationSpace):
    """N-dimensional full iteration space.

    Each dimension of the full iteration space is represented by an instance of `FullIterationSpace.Dimension`.
    Dimensions are ordered slowest-to-fastest: The first dimension corresponds to the slowest coordinate,
    translates to the outermost loop, while the last dimension is the fastest coordinate and translates
    to the innermost loop.
    """

    @dataclass
    class Dimension:
        """One dimension of a dense iteration space"""

        start: PsExpression
        stop: PsExpression
        step: PsExpression
        counter: PsSymbol

    @staticmethod
    def create_with_ghost_layers(
        ctx: KernelCreationContext,
        ghost_layers: int | Sequence[int | tuple[int, int]],
        archetype_field: Field,
    ) -> FullIterationSpace:
        """Create an iteration space over an archetype field with ghost layers."""

        archetype_array = ctx.get_buffer(archetype_field)
        dim = archetype_field.spatial_dimensions

        counters = [
            ctx.get_symbol(name, ctx.index_dtype)
            for name in DEFAULTS.spatial_counter_names[:dim]
        ]

        if isinstance(ghost_layers, int):
            ghost_layers_spec = [(ghost_layers, ghost_layers) for _ in range(dim)]
        else:
            if len(ghost_layers) != dim:
                raise ValueError("Too few entries in ghost layer spec")
            ghost_layers_spec = [
                ((gl, gl) if isinstance(gl, int) else gl) for gl in ghost_layers
            ]

        one = PsConstantExpr(PsConstant(1, ctx.index_dtype))

        ghost_layer_exprs = [
            (
                PsConstantExpr(PsConstant(gl_left, ctx.index_dtype)),
                PsConstantExpr(PsConstant(gl_right, ctx.index_dtype)),
            )
            for (gl_left, gl_right) in ghost_layers_spec
        ]

        spatial_shape = archetype_array.shape[:dim]

        from .typification import Typifier

        typify = Typifier(ctx)

        dimensions = [
            FullIterationSpace.Dimension(
                gl_left, typify(PsExpression.make(shape) - gl_right), one, ctr
            )
            for (gl_left, gl_right), shape, ctr in zip(
                ghost_layer_exprs, spatial_shape, counters, strict=True
            )
        ]

        return FullIterationSpace(ctx, dimensions, archetype_field=archetype_field)

    @staticmethod
    def create_from_slice(
        ctx: KernelCreationContext,
        iteration_slice: int | slice | tuple[int | slice, ...],
        archetype_field: Field | None = None,
    ):
        """Create an iteration space from a sequence of slices, optionally over an archetype field.

        Args:
            ctx: The kernel creation context
            iteration_slice: The iteration slices for each dimension; for valid formats, see `AstFactory.parse_slice`
            archetype_field: Optionally, an archetype field that dictates the upper slice limits and loop order.
        """
        if not isinstance(iteration_slice, tuple):
            iteration_slice = (iteration_slice,)

        dim = len(iteration_slice)
        if dim == 0:
            raise ValueError(
                "At least one slice must be specified to create an iteration space"
            )

        archetype_size: tuple[PsSymbol | PsConstant | None, ...]
        if archetype_field is not None:
            archetype_array = ctx.get_buffer(archetype_field)

            if archetype_field.spatial_dimensions != dim:
                raise ValueError(
                    f"Number of dimensions in slice ({len(iteration_slice)}) "
                    f" did not equal iteration space dimensionality ({dim})"
                )

            archetype_size = tuple(archetype_array.shape[:dim])
        else:
            archetype_size = (None,) * dim

        counters = [
            ctx.get_symbol(name, ctx.index_dtype)
            for name in DEFAULTS.spatial_counter_names[:dim]
        ]

        from .ast_factory import AstFactory

        factory = AstFactory(ctx)

        def to_dim(
            slic: int | slice, size: PsSymbol | PsConstant | None, ctr: PsSymbol
        ):
            start, stop, step = factory.parse_slice(slic, size)
            return FullIterationSpace.Dimension(start, stop, step, ctr)

        dimensions = [
            to_dim(slic, size, ctr)
            for slic, size, ctr in zip(
                iteration_slice, archetype_size, counters, strict=True
            )
        ]

        return FullIterationSpace(ctx, dimensions, archetype_field=archetype_field)

    def __init__(
        self,
        ctx: KernelCreationContext,
        dimensions: Sequence[FullIterationSpace.Dimension],
        archetype_field: Field | None = None,
    ):
        super().__init__(tuple(dim.counter for dim in dimensions))

        self._ctx = ctx
        self._dimensions = dimensions

        self._archetype_field = archetype_field

    @property
    def dimensions(self):
        """The dimensions of this iteration space"""
        return self._dimensions

    @property
    def counters(self) -> tuple[PsSymbol, ...]:
        return tuple(dim.counter for dim in self._dimensions)

    @property
    def lower(self) -> tuple[PsExpression, ...]:
        """Lower limits of each dimension"""
        return tuple(dim.start for dim in self._dimensions)

    @property
    def upper(self) -> tuple[PsExpression, ...]:
        """Upper limits of each dimension"""
        return tuple(dim.stop for dim in self._dimensions)

    @property
    def steps(self) -> tuple[PsExpression, ...]:
        """Iteration steps of each dimension"""
        return tuple(dim.step for dim in self._dimensions)

    @property
    def archetype_field(self) -> Field | None:
        """Field whose shape and memory layout act as archetypes for this iteration space's dimensions."""
        return self._archetype_field

    @property
    def loop_order(self) -> tuple[int, ...]:
        """Return the loop order of this iteration space, ordered from slowest to fastest coordinate."""
        if self._archetype_field is not None:
            return self._archetype_field.layout
        else:
            return tuple(range(len(self.dimensions)))

    def dimensions_in_loop_order(self) -> Sequence[FullIterationSpace.Dimension]:
        """Return the dimensions of this iteration space ordered from the slowest to the fastest coordinate.

        If this iteration space has an `archetype field <FullIterationSpace.archetype_field>` set,
        its field layout is used to determine the ideal loop order;
        otherwise, the dimensions are returned as they are
        """
        return [self._dimensions[i] for i in self.loop_order]

    def actual_iterations(
        self, dimension: int | FullIterationSpace.Dimension | None = None
    ) -> PsExpression:
        """Construct an expression representing the actual number of unique points inside the iteration space.

        Args:
            dimension: If an integer or a `Dimension` object is given, the number of iterations in that
                dimension is computed. If `None`, the total number of iterations inside the entire space
                is computed.
        """
        from .typification import Typifier
        from ..transformations import EliminateConstants

        typify = Typifier(self._ctx)
        fold = EliminateConstants(self._ctx)

        if dimension is None:
            return fold(
                typify(
                    reduce(
                        mul,
                        (
                            self.actual_iterations(d)
                            for d in range(len(self.dimensions))
                        ),
                    )
                )
            )
        else:
            if isinstance(dimension, FullIterationSpace.Dimension):
                dim = dimension
            else:
                dim = self.dimensions[dimension]
            one = PsConstantExpr(PsConstant(1, self._ctx.index_dtype))
            zero = PsConstantExpr(PsConstant(0, self._ctx.index_dtype))
            return fold(
                typify(
                    PsTernary(
                        PsEq(PsRem((dim.stop - dim.start), dim.step), zero),
                        (dim.stop - dim.start) / dim.step,
                        (dim.stop - dim.start) / dim.step + one,
                    )
                )
            )

    def compressed_counter(self) -> PsExpression:
        """Expression counting the actual number of items processed at the iteration defined by the counter tuple.

        Used primarily for indexing buffers."""
        actual_iters = [self.actual_iterations(d) for d in range(self.rank)]
        compressed_counters = [
            (PsExpression.make(dim.counter) - dim.start) / dim.step
            for dim in self.dimensions
        ]
        compressed_idx = compressed_counters[0]
        for ctr, iters in zip(compressed_counters[1:], actual_iters[1:]):
            compressed_idx = compressed_idx * iters + ctr
        return compressed_idx


class SparseIterationSpace(IterationSpace):
    """Represents a sparse iteration space defined by an index list."""

    def __init__(
        self,
        spatial_indices: Sequence[PsSymbol],
        index_list: PsBuffer,
        coordinate_members: Sequence[PsStructType.Member],
        sparse_counter: PsSymbol,
    ):
        super().__init__(spatial_indices)
        self._index_list = index_list
        self._coord_members = tuple(coordinate_members)
        self._sparse_counter = sparse_counter

    @property
    def index_list(self) -> PsBuffer:
        return self._index_list

    @property
    def coordinate_members(self) -> tuple[PsStructType.Member, ...]:
        return self._coord_members

    @property
    def sparse_counter(self) -> PsSymbol:
        return self._sparse_counter


def get_archetype_field(
    fields: set[Field],
    check_compatible_shapes: bool = True,
    check_same_layouts: bool = True,
    check_same_dimensions: bool = True,
):
    """Retrieve an archetype field from a collection of fields, which represents their common properties.

    Raises:
        KernelConstrainsError: If any two fields with conflicting properties are encountered.
    """

    shapes = set(f.spatial_shape for f in fields)
    fixed_shapes = set(f.spatial_shape for f in fields if f.has_fixed_shape)
    layouts = set(f.layout for f in fields)
    dimensionalities = set(f.spatial_dimensions for f in fields)

    if check_same_dimensions and len(dimensionalities) != 1:
        raise KernelConstraintsError(
            "All fields must have the same number of spatial dimensions."
        )

    if check_same_layouts and len(layouts) != 1:
        raise KernelConstraintsError("All fields must have the same memory layout.")

    if check_compatible_shapes:
        if len(fixed_shapes) > 0:
            if len(fixed_shapes) != len(shapes):
                raise KernelConstraintsError(
                    "Cannot mix fixed- and variable-shape fields."
                )
            if len(fixed_shapes) > 1:
                raise KernelConstraintsError(
                    "Fixed-shape fields of different sizes encountered."
                )

    archetype_field = sorted(fields, key=lambda f: str(f))[0]
    return archetype_field


def create_sparse_iteration_space(
    ctx: KernelCreationContext,
    assignments: AssignmentCollection,
    index_field: Field | None = None,
) -> IterationSpace:
    #   All domain and custom fields must have the same spatial dimensions
    #   TODO: Must all domain fields have the same shape?
    archetype_field = get_archetype_field(
        ctx.fields.domain_fields | ctx.fields.custom_fields,
        check_compatible_shapes=False,
        check_same_layouts=False,
        check_same_dimensions=True,
    )

    dim = archetype_field.spatial_dimensions
    coord_members = [
        PsStructType.Member(name, ctx.index_dtype)
        for name in DEFAULTS.index_struct_coordinate_names[:dim]
    ]

    #   Determine index field
    if index_field is not None:
        idx_arr = ctx.get_buffer(index_field)
        idx_struct_type: PsStructType = failing_cast(PsStructType, idx_arr.element_type)

        for coord in coord_members:
            if coord not in idx_struct_type.members:
                raise PsInputError(
                    f"Given index field does not provide required coordinate member {coord}"
                )
    else:
        #   TODO: Find index field from the fields list
        raise NotImplementedError(
            "Automatic inference of index field for sparse iteration not supported yet."
        )

    spatial_counters = [
        ctx.get_symbol(name, ctx.index_dtype)
        for name in DEFAULTS.spatial_counter_names[:dim]
    ]

    sparse_counter = ctx.get_symbol(DEFAULTS.sparse_counter_name, ctx.index_dtype)

    return SparseIterationSpace(
        spatial_counters, idx_arr, coord_members, sparse_counter
    )


def create_full_iteration_space(
    ctx: KernelCreationContext,
    assignments: AssignmentCollection,
    ghost_layers: None | int | Sequence[int | tuple[int, int]] = None,
    iteration_slice: None | int | slice | tuple[int | slice, ...] = None,
    infer_ghost_layers: bool = False,
) -> IterationSpace:
    """Create a dense iteration space from a sequence of assignments and iteration slice information.

    This function finds all accesses to fields in the given assignment collection,
    analyzes the set of fields involved,
    and determines the iteration space bounds from these.
    This requires that either all fields are of the same, fixed, shape, or all of them are
    variable-shaped.
    Also, all fields need to have the same memory layout of their spatial dimensions.

    Args:
        ctx: The kernel creation context
        assignments: Collection of assignments the iteration space should be inferred from
        ghost_layers: If set, strip off that many ghost layers from all sides of the iteration cuboid
        iteration_slice: If set, constrain iteration to the given slice.
            For details on the parsing of slices, see `AstFactory.parse_slice`.
        infer_ghost_layers: If `True`, infer the number of ghost layers from the stencil ranges
            used in the kernel.

    Returns:
        IterationSpace: The constructed iteration space.

    Raises:
        KernelConstraintsError: If field shape or memory layout conflicts are detected
        ValueError: If the iteration slice could not be parsed

    .. attention::
        The ``ghost_layers`` and ``iteration_slice`` arguments are mutually exclusive.
        Also, if ``infer_ghost_layers=True``, none of them may be set.
    """

    assert not ctx.fields.index_fields

    if (ghost_layers is None) and (iteration_slice is None) and not infer_ghost_layers:
        raise ValueError(
            "One argument of `ghost_layers`, `iteration_slice`, and `infer_ghost_layers` must be set."
        )

    if (
        int(ghost_layers is not None)
        + int(iteration_slice is not None)
        + int(infer_ghost_layers)
        > 1
    ):
        raise ValueError(
            "At most one of `ghost_layers`, `iteration_slice`, and `infer_ghost_layers` may be set."
        )

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

    if len(domain_field_accesses) > 0:
        archetype_field = get_archetype_field(ctx.fields.domain_fields)
    elif len(ctx.fields.custom_fields) > 0:
        #   TODO: Warn about inferring iteration space from custom fields
        archetype_field = get_archetype_field(ctx.fields.custom_fields)
    else:
        raise PsInputError(
            "Unable to construct iteration space: The kernel contains no accesses to domain or custom fields."
        )

    # If the user provided a ghost layer specification, use that
    # Otherwise, if an iteration slice was specified, use that
    # Otherwise, use the inferred ghost layers

    if infer_ghost_layers:
        if len(domain_field_accesses) > 0:
            inferred_gls = max(
                [fa.required_ghost_layers for fa in domain_field_accesses]
            )
        else:
            inferred_gls = 0

        ctx.metadata["ghost_layers"] = inferred_gls
        return FullIterationSpace.create_with_ghost_layers(
            ctx, inferred_gls, archetype_field
        )
    elif ghost_layers is not None:
        ctx.metadata["ghost_layers"] = ghost_layers
        return FullIterationSpace.create_with_ghost_layers(
            ctx, ghost_layers, archetype_field
        )
    elif iteration_slice is not None:
        return FullIterationSpace.create_from_slice(
            ctx, iteration_slice, archetype_field
        )
    else:
        assert False, "unreachable code"
