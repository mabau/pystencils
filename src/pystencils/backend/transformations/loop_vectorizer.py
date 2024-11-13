import numpy as np
from enum import Enum, auto
from typing import cast, Callable, overload

from ...types import PsVectorType, PsScalarType

from ..kernelcreation import KernelCreationContext
from ..constants import PsConstant
from ..ast import PsAstNode
from ..ast.structural import PsLoop, PsBlock, PsDeclaration
from ..ast.expressions import PsExpression
from ..ast.vector import PsVecBroadcast
from ..ast.analysis import collect_undefined_symbols

from .ast_vectorizer import VectorizationAxis, VectorizationContext, AstVectorizer
from .rewrite import substitute_symbols


class LoopVectorizer:
    """Vectorize loops.
    
    The loop vectorizer provides methods to vectorize single loops inside an AST
    using a given number of vector lanes.
    During vectorization, the loop body is transformed using the `AstVectorizer`,
    The loop's limits are adapted according to the number of vector lanes,
    and a block treating trailing iterations is optionally added.

    Args:
        ctx: The current kernel creation context
        lanes: The number of vector lanes to use
        trailing_iters: Mode for the treatment of trailing iterations
    """

    class TrailingItersTreatment(Enum):
        """How to treat trailing iterations during loop vectorization."""

        SCALAR_LOOP = auto()
        """Cover trailing iterations using a scalar remainder loop."""

        MASKED_BLOCK = auto()
        """Cover trailing iterations using a masked block."""

        NONE = auto()
        """Assume that the loop iteration count is a multiple of the number of lanes
        and do not cover any trailing iterations"""

    def __init__(
        self,
        ctx: KernelCreationContext,
        lanes: int,
        trailing_iters: TrailingItersTreatment = TrailingItersTreatment.SCALAR_LOOP,
    ):
        self._ctx = ctx
        self._lanes = lanes
        self._trailing_iters = trailing_iters

        from ..kernelcreation import Typifier
        from .eliminate_constants import EliminateConstants

        self._typify = Typifier(ctx)
        self._vectorize_ast = AstVectorizer(ctx)
        self._fold = EliminateConstants(ctx)

    @overload
    def vectorize_select_loops(
        self, node: PsBlock, predicate: Callable[[PsLoop], bool]
    ) -> PsBlock:
        ...

    @overload
    def vectorize_select_loops(
        self, node: PsLoop, predicate: Callable[[PsLoop], bool]
    ) -> PsLoop | PsBlock:
        ...

    @overload
    def vectorize_select_loops(
        self, node: PsAstNode, predicate: Callable[[PsLoop], bool]
    ) -> PsAstNode:
        ...

    def vectorize_select_loops(
        self, node: PsAstNode, predicate: Callable[[PsLoop], bool]
    ) -> PsAstNode:
        """Select and vectorize loops from a syntax tree according to a predicate.
        
        Finds each loop inside a subtree and evaluates ``predicate`` on them.
        If ``predicate(loop)`` evaluates to `True`, the loop is vectorized.
        
        Loops nested inside a vectorized loop will not be processed.

        Args:
            node: Root of the subtree to process
            predicate: Callback telling the vectorizer which loops to vectorize
        """
        match node:
            case PsLoop() if predicate(node):
                return self.vectorize_loop(node)
            case PsExpression():
                return node
            case _:
                node.children = [
                    self.vectorize_select_loops(c, predicate) for c in node.children
                ]
                return node

    def __call__(self, loop: PsLoop) -> PsLoop | PsBlock:
        return self.vectorize_loop(loop)

    def vectorize_loop(self, loop: PsLoop) -> PsLoop | PsBlock:
        """Vectorize the given loop."""
        scalar_ctr_expr = loop.counter
        scalar_ctr = scalar_ctr_expr.symbol

        #   Prepare vector counter
        vector_ctr_dtype = PsVectorType(
            cast(PsScalarType, scalar_ctr_expr.get_dtype()), self._lanes
        )
        vector_ctr = self._ctx.duplicate_symbol(scalar_ctr, vector_ctr_dtype)
        step_multiplier_val = np.array(
            range(self._lanes), dtype=scalar_ctr_expr.get_dtype().numpy_dtype
        )
        step_multiplier = PsExpression.make(
            PsConstant(step_multiplier_val, vector_ctr_dtype)
        )
        vector_counter_decl = self._type_fold(
            PsDeclaration(
                PsExpression.make(vector_ctr),
                PsVecBroadcast(self._lanes, scalar_ctr_expr)
                + step_multiplier * PsVecBroadcast(self._lanes, loop.step),
            )
        )

        #   Prepare axis
        axis = VectorizationAxis(scalar_ctr, vector_ctr, step=loop.step)

        #   Prepare vectorization context
        vc = VectorizationContext(self._ctx, self._lanes, axis)

        #   Generate vectorized loop body
        simd_body = self._vectorize_ast(loop.body, vc)
        
        if vector_ctr in collect_undefined_symbols(simd_body):
            simd_body.statements.insert(0, vector_counter_decl)

        #   Build new loop limits
        simd_start = loop.start.clone()

        simd_step = self._ctx.get_new_symbol(
            f"__{scalar_ctr.name}_simd_step", scalar_ctr.get_dtype()
        )
        simd_step_decl = self._type_fold(
            PsDeclaration(
                PsExpression.make(simd_step),
                loop.step.clone() * PsExpression.make(PsConstant(self._lanes)),
            )
        )

        #   Each iteration must satisfy `ctr + step * (lanes - 1) < stop`
        simd_stop = self._ctx.get_new_symbol(
            f"__{scalar_ctr.name}_simd_stop", scalar_ctr.get_dtype()
        )
        simd_stop_decl = self._type_fold(
            PsDeclaration(
                PsExpression.make(simd_stop),
                loop.stop.clone()
                - (
                    PsExpression.make(PsConstant(self._lanes))
                    - PsExpression.make(PsConstant(1))
                )
                * loop.step.clone(),
            )
        )

        simd_loop = PsLoop(
            PsExpression.make(scalar_ctr),
            simd_start,
            PsExpression.make(simd_stop),
            PsExpression.make(simd_step),
            simd_body,
        )

        #   Treat trailing iterations
        match self._trailing_iters:
            case LoopVectorizer.TrailingItersTreatment.SCALAR_LOOP:
                trailing_start = self._ctx.get_new_symbol(
                    f"__{scalar_ctr.name}_trailing_start", scalar_ctr.get_dtype()
                )
                trailing_start_decl = self._type_fold(
                    PsDeclaration(
                        PsExpression.make(trailing_start),
                        (
                            (
                                PsExpression.make(simd_stop)
                                - simd_start.clone()
                                - PsExpression.make(PsConstant(1))
                            )
                            / PsExpression.make(simd_step)
                            + PsExpression.make(PsConstant(1))
                        )
                        * PsExpression.make(simd_step)
                        + simd_start.clone(),
                    )
                )

                trailing_ctr = self._ctx.duplicate_symbol(scalar_ctr)
                trailing_loop_body = substitute_symbols(
                    loop.body.clone(), {scalar_ctr: PsExpression.make(trailing_ctr)}
                )
                trailing_loop = PsLoop(
                    PsExpression.make(trailing_ctr),
                    PsExpression.make(trailing_start),
                    loop.stop.clone(),
                    loop.step.clone(),
                    trailing_loop_body,
                )

                return PsBlock(
                    [
                        simd_stop_decl,
                        simd_step_decl,
                        simd_loop,
                        trailing_start_decl,
                        trailing_loop,
                    ]
                )

            case LoopVectorizer.TrailingItersTreatment.MASKED_BLOCK:
                raise NotImplementedError()

            case LoopVectorizer.TrailingItersTreatment.NONE:
                return PsBlock(
                    [
                        simd_stop_decl,
                        simd_step_decl,
                        simd_loop,
                    ]
                )

    @overload
    def _type_fold(self, node: PsExpression) -> PsExpression:
        pass

    @overload
    def _type_fold(self, node: PsDeclaration) -> PsDeclaration:
        pass

    @overload
    def _type_fold(self, node: PsAstNode) -> PsAstNode:
        pass

    def _type_fold(self, node: PsAstNode) -> PsAstNode:
        return self._fold(self._typify(node))
