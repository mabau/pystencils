from typing import Sequence
from abc import ABC, abstractmethod

from pystencils.backend.ast.expressions import PsCall

from ..functions import CFunction, PsMathFunction, MathFunctions
from ...types import PsIntegerType, PsIeeeFloatType

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
    PsBufferAcc,
    PsVectorMemAcc,
    PsLookup,
    PsGe,
    PsLe,
    PsTernary
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
            raise MaterializationError(f"Unknown type of iteration space: {ispace}")

    def select_function(self, call: PsCall) -> PsExpression:
        assert isinstance(call.function, PsMathFunction)
        
        func = call.function.func
        dtype = call.get_dtype()
        arg_types = (dtype,) * func.num_args

        if isinstance(dtype, PsIeeeFloatType) and dtype.width in (32, 64):
            cfunc: CFunction
            match func:
                case (
                    MathFunctions.Exp
                    | MathFunctions.Log
                    | MathFunctions.Sin
                    | MathFunctions.Cos
                    | MathFunctions.Tan
                    | MathFunctions.Sinh
                    | MathFunctions.Cosh
                    | MathFunctions.ASin
                    | MathFunctions.ACos
                    | MathFunctions.ATan
                    | MathFunctions.ATan2
                    | MathFunctions.Pow
                    | MathFunctions.Floor
                    | MathFunctions.Ceil
                ):
                    cfunc = CFunction(func.function_name, arg_types, dtype)
                case MathFunctions.Abs | MathFunctions.Min | MathFunctions.Max:
                    cfunc = CFunction("f" + func.function_name, arg_types, dtype)

            call.function = cfunc
            return call
                
        if isinstance(dtype, PsIntegerType):
            match func:
                case MathFunctions.Abs:
                    zero = PsExpression.make(PsConstant(0, dtype))
                    arg = call.args[0]
                    return PsTernary(PsGe(arg, zero), arg, - arg)
                case MathFunctions.Min:
                    arg1, arg2 = call.args
                    return PsTernary(PsLe(arg1, arg2), arg1, arg2)
                case MathFunctions.Max:
                    arg1, arg2 = call.args
                    return PsTernary(PsGe(arg1, arg2), arg1, arg2)

        raise MaterializationError(
            f"No implementation available for function {func} on data type {dtype}"
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
        factory = AstFactory(self._ctx)

        mappings = [
            PsDeclaration(
                PsSymbolExpr(ctr),
                PsLookup(
                    PsBufferAcc(
                        ispace.index_list.base_pointer,
                        (PsExpression.make(ispace.sparse_counter), factory.parse_index(0)),
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
    def vector_load(self, acc: PsVectorMemAcc) -> PsExpression:
        """Return an expression intrinsically performing a vector load,
        or raise an `MaterializationError` if not supported."""

    @abstractmethod
    def vector_store(self, acc: PsVectorMemAcc, arg: PsExpression) -> PsExpression:
        """Return an expression intrinsically performing a vector store,
        or raise an `MaterializationError` if not supported."""
