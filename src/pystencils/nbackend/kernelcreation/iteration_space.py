from __future__ import annotations
from typing import Sequence, TYPE_CHECKING
from abc import ABC
from dataclasses import dataclass
from functools import reduce
from operator import mul

from ...field import Field

from ..typed_expressions import (
    PsTypedVariable,
    VarOrConstant,
    ExprOrConstant,
    PsTypedConstant,
)
from ..arrays import PsLinearizedArray
from .defaults import Pymbolic as Defaults

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
            for name in Defaults.spatial_counter_names
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

        dimensions = [
            FullIterationSpace.Dimension(gl_left, shape - gl_right, one, ctr)
            for (gl_left, gl_right), shape, ctr in zip(
                ghost_layer_exprs, archetype_array.shape, counters, strict=True
            )
        ]

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
