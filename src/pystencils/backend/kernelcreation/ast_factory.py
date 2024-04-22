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
from .freeze import FreezeExpressions
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
    def parse_sympy(self, sp_obj: sp.Expr) -> PsExpression:
        pass

    @overload
    def parse_sympy(self, sp_obj: AssignmentBase) -> PsAssignment:
        pass

    def parse_sympy(self, sp_obj: sp.Expr | AssignmentBase) -> PsAstNode:
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
        return self.parse_index(cast(IndexParsable, idx))

    def parse_slice(
        self, slic: slice, upper_limit: Any | None = None
    ) -> tuple[PsExpression, PsExpression, PsExpression]:
        """Parse a slice to obtain start, stop and step expressions for a loop or iteration space dimension.

        The slice entries may be instances of `PsExpression`, `PsSymbol` or `PsConstant`, in which case they
        must typify with the kernel creation context's ``index_dtype``.
        They may also be sympy expressions or integer constants, in which case they are parsed to AST objects
        and must also typify with the kernel creation context's ``index_dtype``.

        If the slice's ``stop`` member is `None` or a negative `int`, `upper_limit` must be specified, which is then
        used as the upper iteration limit as either ``upper_limit`` or ``upper_limit - stop``.

        Args:
            slic: The iteration slice
            upper_limit: Optionally, the upper iteration limit
        """

        if slic.stop is None or (isinstance(slic.stop, int) and slic.stop < 0):
            if upper_limit is None:
                raise ValueError(
                    "Must specify an upper iteration limit if `slice.stop` is `None` or a negative `int`"
                )

        start = self._parse_any_index(slic.start if slic.start is not None else 0)
        stop = (
            self._parse_any_index(slic.stop)
            if slic.stop is not None
            else self._parse_any_index(upper_limit)
        )
        step = self._parse_any_index(slic.step if slic.step is not None else 1)

        if isinstance(slic.stop, int) and slic.stop < 0:
            stop = self._parse_any_index(upper_limit) + stop

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
