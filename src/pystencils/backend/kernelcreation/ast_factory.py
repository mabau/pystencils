from typing import Any, Sequence, cast, overload

import numpy as np
import sympy as sp
from sympy.codegen.ast import AssignmentBase

from ..ast import PsAstNode
from ..ast.expressions import PsExpression, PsSymbolExpr, PsConstantExpr
from ..ast.structural import PsLoop, PsBlock, PsAssignment

from ..symbols import PsSymbol
from ..constants import PsConstant

from .context import KernelCreationContext
from .freeze import FreezeExpressions, ExprLike
from .typification import Typifier
from .iteration_space import FullIterationSpace


IndexParsable = PsExpression | PsSymbol | PsConstant | sp.Expr | int | np.integer
_IndexParsable = (PsExpression, PsSymbol, PsConstant, sp.Expr, int, np.integer)


class AstFactory:
    """Factory providing a convenient interface for building syntax trees.

    The `AstFactory` uses the defaults provided by the given `KernelCreationContext` to quickly create
    AST nodes. Depending on context (numerical, loop indexing, etc.), symbols and constants receive either
    ``ctx.default_dtype`` or ``ctx.index_dtype``.

    Args:
        ctx: The kernel creation context
    """

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx
        self._freeze = FreezeExpressions(ctx)
        self._typify = Typifier(ctx)

    @overload
    def parse_sympy(self, sp_obj: sp.Symbol) -> PsSymbolExpr:
        pass

    @overload
    def parse_sympy(self, sp_obj: ExprLike) -> PsExpression:
        pass

    @overload
    def parse_sympy(self, sp_obj: AssignmentBase) -> PsAssignment:
        pass

    def parse_sympy(self, sp_obj: ExprLike | AssignmentBase) -> PsAstNode:
        """Parse a SymPy expression or assignment through `FreezeExpressions` and `Typifier`.

        The expression or assignment will be typified in a numerical context, using the kernel
        creation context's `default_dtype`.

        Args:
            sp_obj: A SymPy expression or assignment
        """
        return self._typify(self._freeze(sp_obj))

    @overload
    def parse_index(self, idx: sp.Symbol | PsSymbol | PsSymbolExpr) -> PsSymbolExpr:
        pass

    @overload
    def parse_index(
        self, idx: int | np.integer | PsConstant | PsConstantExpr
    ) -> PsConstantExpr:
        pass

    @overload
    def parse_index(self, idx: sp.Expr | PsExpression) -> PsExpression:
        pass

    def parse_index(self, idx: IndexParsable):
        """Parse the given object as an expression with data type `ctx.index_dtype`."""

        if not isinstance(idx, _IndexParsable):
            raise TypeError(
                f"Cannot parse object of type {type(idx)} as an index expression"
            )

        match idx:
            case PsExpression():
                return self._typify.typify_expression(idx, self._ctx.index_dtype)[0]
            case PsSymbol() | PsConstant():
                return self._typify.typify_expression(
                    PsExpression.make(idx), self._ctx.index_dtype
                )[0]
            case sp.Expr():
                return self._typify.typify_expression(
                    self._freeze(idx), self._ctx.index_dtype
                )[0]
            case _:
                return PsExpression.make(PsConstant(idx, self._ctx.index_dtype))

    def _parse_any_index(self, idx: Any) -> PsExpression:
        if not isinstance(idx, _IndexParsable):
            raise TypeError(f"Cannot parse {idx} as an index expression")
        return self.parse_index(idx)

    def parse_slice(
        self,
        iter_slice: IndexParsable | slice,
        normalize_to: IndexParsable | None = None,
    ) -> tuple[PsExpression, PsExpression, PsExpression]:
        """Parse a slice to obtain start, stop and step expressions for a loop or iteration space dimension.

        The slice entries may be instances of `PsExpression`, `PsSymbol` or `PsConstant`, in which case they
        must typify with the kernel creation context's ``index_dtype``.
        They may also be sympy expressions or integer constants, in which case they are parsed to AST objects
        and must also typify with the kernel creation context's ``index_dtype``.

        The `step` member of the slice, if it is constant, must be positive.

        The slice may optionally be normalized with respect to an upper iteration limit.
        If `normalize_to` is specified, negative integers in `iter_slice.start` and `iter_slice.stop` will
        be added to that normalization limit.

        Args:
            iter_slice: The iteration slice
            normalize_to: The upper iteration limit with respect to which the slice should be normalized
        """

        from pystencils.backend.transformations import EliminateConstants

        fold = EliminateConstants(self._ctx)

        start: PsExpression
        stop: PsExpression | None
        step: PsExpression

        if not isinstance(iter_slice, slice):
            start = self.parse_index(iter_slice)
            stop = fold(
                self._typify(self.parse_index(iter_slice) + self.parse_index(1))
            )
            step = self.parse_index(1)
        else:
            start = self._parse_any_index(
                iter_slice.start if iter_slice.start is not None else 0
            )
            stop = (
                self._parse_any_index(iter_slice.stop)
                if iter_slice.stop is not None
                else None
            )
            step = self._parse_any_index(
                iter_slice.step if iter_slice.step is not None else 1
            )

            if isinstance(step, PsConstantExpr) and step.constant.value <= 0:
                raise ValueError(
                    f"Invalid value for `slice.step`: {step.constant.value}"
                )

        if normalize_to is not None:
            upper_limit = self.parse_index(normalize_to)
            if isinstance(start, PsConstantExpr) and start.constant.value < 0:
                start = fold(self._typify(upper_limit.clone() + start))

            if stop is None:
                stop = upper_limit
            elif isinstance(stop, PsConstantExpr) and stop.constant.value < 0:
                stop = fold(self._typify(upper_limit.clone() + stop))

        elif stop is None:
            raise ValueError(
                "Cannot parse a slice with `stop == None` if no normalization limit is given"
            )

        return start, stop, step

    def loop(self, ctr_name: str, iteration_slice: slice, body: PsBlock):
        """Create a loop from a slice.

        Args:
            ctr_name: Name of the loop counter
            iteration_slice: The iteration region as a slice; see `parse_slice`.
            body: The loop body
        """
        ctr = PsExpression.make(self._ctx.get_symbol(ctr_name, self._ctx.index_dtype))

        start, stop, step = self.parse_slice(iteration_slice)

        return PsLoop(
            ctr,
            start,
            stop,
            step,
            body,
        )

    def loop_nest(
        self, counters: Sequence[str], slices: Sequence[slice], body: PsBlock
    ) -> PsLoop:
        """Create a loop nest from a sequence of slices.

        **Example:**
        This snippet creates a 3D loop nest with ten iterations in each dimension::

        >>> from pystencils import make_slice
        >>> ctx = KernelCreationContext()
        >>> factory = AstFactory(ctx)
        >>> loop = factory.loop_nest(("i", "j", "k"), make_slice[:10,:10,:10], PsBlock([]))

        Args:
            counters: Sequence of names for the loop counters
            slices: Sequence of iteration slices; see also `parse_slice`
            body: The loop body
        """
        if not slices:
            raise ValueError(
                "At least one slice must be specified to create a loop nest."
            )

        ast = body
        for ctr_name, sl in zip(counters[::-1], slices[::-1], strict=True):
            ast = self.loop(
                ctr_name,
                sl,
                PsBlock([ast]) if not isinstance(ast, PsBlock) else ast,
            )

        return cast(PsLoop, ast)

    def loops_from_ispace(
        self,
        ispace: FullIterationSpace,
        body: PsBlock,
        loop_order: Sequence[int] | None = None,
    ) -> PsLoop:
        """Create a loop nest from a dense iteration space.

        Args:
            ispace: The iteration space object
            body: The loop body
            loop_order: Optionally, a permutation of integers indicating the order of loops
        """
        dimensions = ispace.dimensions

        if loop_order is not None:
            dimensions = [dimensions[coordinate] for coordinate in loop_order]

        outer_node: PsLoop | PsBlock = body

        for dimension in dimensions[::-1]:
            outer_node = PsLoop(
                PsSymbolExpr(dimension.counter),
                dimension.start,
                dimension.stop,
                dimension.step,
                (
                    outer_node
                    if isinstance(outer_node, PsBlock)
                    else PsBlock([outer_node])
                ),
            )

        assert isinstance(outer_node, PsLoop)
        return outer_node
