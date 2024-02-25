from typing import Sequence
from abc import ABC, abstractmethod

import pymbolic.primitives as pb

from .platform import Platform

from ..kernelcreation.iteration_space import (
    IterationSpace,
    FullIterationSpace,
    SparseIterationSpace,
)

from ..ast import PsDeclaration, PsSymbolExpr, PsExpression, PsLoop, PsBlock
from ..types import PsVectorType, PsCustomType
from ..typed_expressions import PsTypedConstant
from ..arrays import PsArrayAccess, PsVectorArrayAccess
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

    def optimize(self, kernel: PsBlock) -> PsBlock:
        return kernel

    #   Internals

    def _create_domain_loops(
        self, body: PsBlock, ispace: FullIterationSpace
    ) -> PsBlock:
        dimensions = ispace.dimensions
        outer_block = body

        for dimension in dimensions[::-1]:
            loop = PsLoop(
                PsSymbolExpr(dimension.counter),
                PsExpression(dimension.start),
                PsExpression(dimension.stop),
                PsExpression(dimension.step),
                outer_block,
            )
            outer_block = PsBlock([loop])

        return outer_block

    def _create_sparse_loop(self, body: PsBlock, ispace: SparseIterationSpace):
        mappings = [
            PsDeclaration(
                PsSymbolExpr(ctr),
                PsExpression(
                    PsArrayAccess(
                        ispace.index_list.base_pointer, ispace.sparse_counter
                    ).a.__getattr__(coord.name)
                ),
            )
            for ctr, coord in zip(ispace.spatial_indices, ispace.coordinate_members)
        ]

        body = PsBlock(mappings + body.statements)

        loop = PsLoop(
            PsSymbolExpr(ispace.sparse_counter),
            PsExpression(PsTypedConstant(0, self._ctx.index_dtype)),
            PsExpression(ispace.index_list.shape[0]),
            PsExpression(PsTypedConstant(1, self._ctx.index_dtype)),
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
    def constant_vector(self, c: PsTypedConstant) -> pb.Expression:
        """Return an expression that initializes a constant vector,
        or raise an `IntrinsicsError` if not supported."""

    @abstractmethod
    def op_intrinsic(
        self, op: IntrinsicOps, vtype: PsVectorType, args: Sequence[pb.Expression]
    ) -> pb.Expression:
        """Return an expression intrinsically invoking the given operation
        on the given arguments with the given vector type,
        or raise an `IntrinsicsError` if not supported."""

    @abstractmethod
    def vector_load(self, acc: PsVectorArrayAccess) -> pb.Expression:
        """Return an expression intrinsically performing a vector load,
        or raise an `IntrinsicsError` if not supported."""

    @abstractmethod
    def vector_store(
        self, acc: PsVectorArrayAccess, arg: pb.Expression
    ) -> pb.Expression:
        """Return an expression intrinsically performing a vector store,
        or raise an `IntrinsicsError` if not supported."""
