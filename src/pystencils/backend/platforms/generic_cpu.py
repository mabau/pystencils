from typing import Sequence
from abc import ABC, abstractmethod

from ..functions import CFunction, PsMathFunction, MathFunctions
from ...types import PsType, PsIeeeFloatType

from .platform import Platform
from ..exceptions import MaterializationError

from ..kernelcreation import AstFactory
from ..kernelcreation.iteration_space import (
    IterationSpace,
    FullIterationSpace,
    SparseIterationSpace,
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
from ..transformations.select_intrinsics import IntrinsicOps


class GenericCpu(Platform):
    """Generic CPU platform.

    The `GenericCPU` platform models the following execution environment:

     - Generic multicore CPU architecture
     - Iteration space represented by a loop nest, kernels are executed as a whole
     - C standard library math functions available (``#include <math.h>`` or ``#include <cmath>``)
    """

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

    def select_function(
        self, math_function: PsMathFunction, dtype: PsType
    ) -> CFunction:
        func = math_function.func
        if isinstance(dtype, PsIeeeFloatType) and dtype.width in (32, 64):
            match func:
                case (
                    MathFunctions.Exp
                    | MathFunctions.Sin
                    | MathFunctions.Cos
                    | MathFunctions.Tan
                    | MathFunctions.Pow
                ):
                    return CFunction(func.function_name, func.num_args)
                case MathFunctions.Abs | MathFunctions.Min | MathFunctions.Max:
                    return CFunction("f" + func.function_name, func.num_args)

        raise MaterializationError(
            f"No implementation available for function {math_function} on data type {dtype}"
        )

    #   Internals

    def _create_domain_loops(
        self, body: PsBlock, ispace: FullIterationSpace
    ) -> PsBlock:
        factory = AstFactory(self._ctx)

        #   Determine loop order by permuting dimensions
        archetype_field = ispace.archetype_field
        if archetype_field is not None:
            loop_order = archetype_field.layout
        else:
            loop_order = None

        loops = factory.loops_from_ispace(ispace, body, loop_order)
        return PsBlock([loops])

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


class GenericVectorCpu(GenericCpu, ABC):
    """Base class for CPU platforms with vectorization support through intrinsics."""

    @abstractmethod
    def type_intrinsic(self, vector_type: PsVectorType) -> PsCustomType:
        """Return the intrinsic vector type for the given generic vector type,
        or raise an `MaterializationError` if type is not supported."""

    @abstractmethod
    def constant_vector(self, c: PsConstant) -> PsExpression:
        """Return an expression that initializes a constant vector,
        or raise an `MaterializationError` if not supported."""

    @abstractmethod
    def op_intrinsic(
        self, op: IntrinsicOps, vtype: PsVectorType, args: Sequence[PsExpression]
    ) -> PsExpression:
        """Return an expression intrinsically invoking the given operation
        on the given arguments with the given vector type,
        or raise an `MaterializationError` if not supported."""

    @abstractmethod
    def vector_load(self, acc: PsVectorArrayAccess) -> PsExpression:
        """Return an expression intrinsically performing a vector load,
        or raise an `MaterializationError` if not supported."""

    @abstractmethod
    def vector_store(self, acc: PsVectorArrayAccess, arg: PsExpression) -> PsExpression:
        """Return an expression intrinsically performing a vector store,
        or raise an `MaterializationError` if not supported."""
