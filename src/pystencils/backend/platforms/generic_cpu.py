from typing import Sequence
from abc import ABC, abstractmethod

from .platform import Platform

from ..kernelcreation.iteration_space import (
    IterationSpace,
    FullIterationSpace,
    SparseIterationSpace
)

from ..constants import PsConstant
from ..ast.structural import PsDeclaration, PsLoop, PsBlock
from ..ast.expressions import (
    PsSymbolExpr,
    PsExpression,
    PsArrayAccess,
    PsVectorArrayAccess,
    PsLookup,
)
from ...types import PsVectorType, PsCustomType
from ..transformations.vector_intrinsics import IntrinsicOps


class GenericCpu(Platform):

    @property
    def required_headers(self) -> set[str]:
        return {"<math.h>"}

    def materialize_iteration_space(
        self, body: PsBlock, ispace: IterationSpace
    ) -> PsBlock:
        if isinstance(ispace, FullIterationSpace):
            return self._create_domain_loops(body, ispace)
        elif isinstance(ispace, SparseIterationSpace):
            return self._create_sparse_loop(body, ispace)
        else:
            assert False, "unreachable code"

    #   Internals

    def _create_domain_loops(
        self, body: PsBlock, ispace: FullIterationSpace
    ) -> PsBlock:
        
        dimensions = ispace.dimensions

        #   Determine loop order by permuting dimensions
        archetype_field = ispace.archetype_field
        if archetype_field is not None:
            loop_order = archetype_field.layout
            dimensions = [dimensions[coordinate] for coordinate in loop_order]

        outer_block = body

        for dimension in dimensions[::-1]:
            loop = PsLoop(
                PsSymbolExpr(dimension.counter),
                dimension.start,
                dimension.stop,
                dimension.step,
                outer_block,
            )
            outer_block = PsBlock([loop])

        return outer_block

    def _create_sparse_loop(self, body: PsBlock, ispace: SparseIterationSpace):
        mappings = [
            PsDeclaration(
                PsSymbolExpr(ctr),
                PsLookup(
                    PsArrayAccess(
                        ispace.index_list.base_pointer,
                        PsExpression.make(ispace.sparse_counter),
                    ),
                    coord.name,
                ),
            )
            for ctr, coord in zip(ispace.spatial_indices, ispace.coordinate_members)
        ]

        body = PsBlock(mappings + body.statements)

        loop = PsLoop(
            PsSymbolExpr(ispace.sparse_counter),
            PsExpression.make(PsConstant(0, self._ctx.index_dtype)),
            PsExpression.make(ispace.index_list.shape[0]),
            PsExpression.make(PsConstant(1, self._ctx.index_dtype)),
            body,
        )

        return PsBlock([loop])


class IntrinsicsError(Exception):
    """Exception indicating a fatal error during intrinsic materialization."""


class GenericVectorCpu(GenericCpu, ABC):

    @abstractmethod
    def type_intrinsic(self, vector_type: PsVectorType) -> PsCustomType:
        """Return the intrinsic vector type for the given generic vector type,
        or raise an `IntrinsicsError` if type is not supported."""

    @abstractmethod
    def constant_vector(self, c: PsConstant) -> PsExpression:
        """Return an expression that initializes a constant vector,
        or raise an `IntrinsicsError` if not supported."""

    @abstractmethod
    def op_intrinsic(
        self, op: IntrinsicOps, vtype: PsVectorType, args: Sequence[PsExpression]
    ) -> PsExpression:
        """Return an expression intrinsically invoking the given operation
        on the given arguments with the given vector type,
        or raise an `IntrinsicsError` if not supported."""

    @abstractmethod
    def vector_load(self, acc: PsVectorArrayAccess) -> PsExpression:
        """Return an expression intrinsically performing a vector load,
        or raise an `IntrinsicsError` if not supported."""

    @abstractmethod
    def vector_store(self, acc: PsVectorArrayAccess, arg: PsExpression) -> PsExpression:
        """Return an expression intrinsically performing a vector store,
        or raise an `IntrinsicsError` if not supported."""
