from dataclasses import dataclass
from typing import cast
from functools import reduce
import operator

from .structural import (
    PsAssignment,
    PsAstNode,
    PsBlock,
    PsEmptyLeafMixIn,
    PsConditional,
    PsDeclaration,
    PsExpression,
    PsLoop,
    PsStatement,
)
from .expressions import (
    PsAdd,
    PsBufferAcc,
    PsCall,
    PsConstantExpr,
    PsDiv,
    PsIntDiv,
    PsLiteralExpr,
    PsMul,
    PsNeg,
    PsRem,
    PsSub,
    PsSymbolExpr,
    PsTernary,
    PsSubscript,
    PsMemAcc,
)

from ..memory import PsSymbol
from ..exceptions import PsInternalCompilerError

from ...types import PsNumericType
from ...types.exception import PsTypeError


class UndefinedSymbolsCollector:
    """Collect undefined symbols.

    This class implements an AST visitor that collects all symbols that have been used
    in the AST without being defined prior to their usage.
    """

    def __call__(self, node: PsAstNode) -> set[PsSymbol]:
        """Returns all symbols that occur in the given AST without being defined prior to their usage."""
        return self.visit(node)

    def visit(self, node: PsAstNode) -> set[PsSymbol]:
        undefined_vars: set[PsSymbol] = set()

        match node:
            case PsExpression():
                return self.visit_expr(node)

            case PsStatement(expr):
                return self.visit_expr(expr)

            case PsAssignment(lhs, rhs):
                undefined_vars = self(lhs) | self(rhs)
                if isinstance(lhs, PsSymbolExpr):
                    undefined_vars.remove(lhs.symbol)
                return undefined_vars

            case PsBlock(statements):
                for stmt in statements[::-1]:
                    undefined_vars -= self.declared_variables(stmt)
                    undefined_vars |= self(stmt)

                return undefined_vars

            case PsLoop(ctr, start, stop, step, body):
                undefined_vars = self(start) | self(stop) | self(step) | self(body)
                undefined_vars.discard(ctr.symbol)
                return undefined_vars

            case PsConditional(cond, branch_true, branch_false):
                undefined_vars = self(cond) | self(branch_true)
                if branch_false is not None:
                    undefined_vars |= self(branch_false)
                return undefined_vars

            case PsEmptyLeafMixIn():
                return set()

            case unknown:
                raise PsInternalCompilerError(
                    f"Don't know how to collect undefined variables from {unknown}"
                )

    def visit_expr(self, expr: PsExpression) -> set[PsSymbol]:
        match expr:
            case PsSymbolExpr(symb):
                return {symb}
            case _:
                return reduce(
                    set.union,
                    (self.visit_expr(cast(PsExpression, c)) for c in expr.children),
                    set(),
                )

    def declared_variables(self, node: PsAstNode) -> set[PsSymbol]:
        """Returns the set of variables declared by the given node which are visible in the enclosing scope."""

        match node:
            case PsDeclaration():
                return {node.declared_symbol}

            case (
                PsAssignment()
                | PsBlock()
                | PsConditional()
                | PsExpression()
                | PsLoop()
                | PsStatement()
                | PsEmptyLeafMixIn()
            ):
                return set()

            case unknown:
                raise PsInternalCompilerError(
                    f"Don't know how to collect declared variables from {unknown}"
                )


def collect_undefined_symbols(node: PsAstNode) -> set[PsSymbol]:
    return UndefinedSymbolsCollector()(node)


def collect_required_headers(node: PsAstNode) -> set[str]:
    match node:
        case PsSymbolExpr(symb):
            return symb.get_dtype().required_headers
        case PsConstantExpr(cs):
            return cs.get_dtype().required_headers
        case _:
            return reduce(
                set.union, (collect_required_headers(c) for c in node.children), set()
            )


@dataclass
class OperationCounts:
    float_adds: int = 0
    float_muls: int = 0
    float_divs: int = 0
    int_adds: int = 0
    int_muls: int = 0
    int_divs: int = 0
    calls: int = 0
    branches: int = 0
    loops_with_dynamic_bounds: int = 0

    def __add__(self, other):
        if not isinstance(other, OperationCounts):
            return NotImplemented

        return OperationCounts(
            float_adds=self.float_adds + other.float_adds,
            float_muls=self.float_muls + other.float_muls,
            float_divs=self.float_divs + other.float_divs,
            int_adds=self.int_adds + other.int_adds,
            int_muls=self.int_muls + other.int_muls,
            int_divs=self.int_divs + other.int_divs,
            calls=self.calls + other.calls,
            branches=self.branches + other.branches,
            loops_with_dynamic_bounds=self.loops_with_dynamic_bounds
            + other.loops_with_dynamic_bounds,
        )

    def __rmul__(self, other):
        if not isinstance(other, int):
            return NotImplemented

        return OperationCounts(
            float_adds=other * self.float_adds,
            float_muls=other * self.float_muls,
            float_divs=other * self.float_divs,
            int_adds=other * self.int_adds,
            int_muls=other * self.int_muls,
            int_divs=other * self.int_divs,
            calls=other * self.calls,
            branches=other * self.branches,
            loops_with_dynamic_bounds=other * self.loops_with_dynamic_bounds,
        )


class OperationCounter:
    """Counts the number of operations in an AST.

    Assumes that the AST is typed. It is recommended that constant folding is
    applied prior to this pass.

    The counted operations are:
      - Additions, multiplications and divisions of floating and integer type.
        The counts of either type are reported separately and operations on
        other types are ignored.
      - Function calls.
      - Branches.
        Includes `PsConditional` and `PsTernary`. The operations in all branches
        are summed up (i.e. the result is an overestimation).
      - Loops with an unknown number of iterations.
        The operations in the loop header and body are counted exactly once,
        i.e. it is assumed that there is one loop iteration.

    If the start, stop and step of the loop are `PsConstantExpr`, then any
    operation within the body is multiplied by the number of iterations.
    """

    def __call__(self, node: PsAstNode) -> OperationCounts:
        """Counts the number of operations in the given AST."""
        return self.visit(node)

    def visit(self, node: PsAstNode) -> OperationCounts:
        match node:
            case PsExpression():
                return self.visit_expr(node)

            case PsStatement(expr):
                return self.visit_expr(expr)

            case PsAssignment(lhs, rhs):
                return self.visit_expr(lhs) + self.visit_expr(rhs)

            case PsBlock(statements):
                return reduce(
                    operator.add, (self.visit(s) for s in statements), OperationCounts()
                )

            case PsLoop(_, start, stop, step, body):
                if (
                    isinstance(start, PsConstantExpr)
                    and isinstance(stop, PsConstantExpr)
                    and isinstance(step, PsConstantExpr)
                ):
                    val_start = start.constant.value
                    val_stop = stop.constant.value
                    val_step = step.constant.value

                    if (val_stop - val_start) % val_step == 0:
                        iteration_count = max(0, int((val_stop - val_start) / val_step))
                    else:
                        iteration_count = max(
                            0, int((val_stop - val_start) / val_step) + 1
                        )

                    return self.visit_expr(start) + iteration_count * (
                        OperationCounts(int_adds=1)  # loop counter increment
                        + self.visit_expr(stop)
                        + self.visit_expr(step)
                        + self.visit(body)
                    )
                else:
                    return (
                        OperationCounts(loops_with_dynamic_bounds=1)
                        + self.visit_expr(start)
                        + self.visit_expr(stop)
                        + self.visit_expr(step)
                        + self.visit(body)
                    )

            case PsConditional(cond, branch_true, branch_false):
                op_counts = (
                    OperationCounts(branches=1)
                    + self.visit(cond)
                    + self.visit(branch_true)
                )
                if branch_false is not None:
                    op_counts += self.visit(branch_false)
                return op_counts

            case PsEmptyLeafMixIn():
                return OperationCounts()

            case unknown:
                raise PsInternalCompilerError(f"Can't count operations in {unknown}")

    def visit_expr(self, expr: PsExpression) -> OperationCounts:
        match expr:
            case PsSymbolExpr(_) | PsConstantExpr(_) | PsLiteralExpr(_):
                return OperationCounts()

            case PsBufferAcc(_, indices) | PsSubscript(_, indices):
                return reduce(operator.add, (self.visit_expr(idx) for idx in indices))

            case PsMemAcc(_, offset):
                return self.visit_expr(offset)

            case PsCall(_, args):
                return OperationCounts(calls=1) + reduce(
                    operator.add, (self.visit(a) for a in args), OperationCounts()
                )

            case PsTernary(cond, then, els):
                return (
                    OperationCounts(branches=1)
                    + self.visit_expr(cond)
                    + self.visit_expr(then)
                    + self.visit_expr(els)
                )

            case PsNeg(arg):
                if expr.dtype is None:
                    raise PsTypeError(f"Untyped arithmetic expression: {expr}")

                op_counts = self.visit_expr(arg)
                if isinstance(expr.dtype, PsNumericType) and expr.dtype.is_float():
                    op_counts.float_muls += 1
                elif isinstance(expr.dtype, PsNumericType) and expr.dtype.is_int():
                    op_counts.int_muls += 1
                return op_counts

            case PsAdd(arg1, arg2) | PsSub(arg1, arg2):
                if expr.dtype is None:
                    raise PsTypeError(f"Untyped arithmetic expression: {expr}")

                op_counts = self.visit_expr(arg1) + self.visit_expr(arg2)
                if isinstance(expr.dtype, PsNumericType) and expr.dtype.is_float():
                    op_counts.float_adds += 1
                elif isinstance(expr.dtype, PsNumericType) and expr.dtype.is_int():
                    op_counts.int_adds += 1
                return op_counts

            case PsMul(arg1, arg2):
                if expr.dtype is None:
                    raise PsTypeError(f"Untyped arithmetic expression: {expr}")

                op_counts = self.visit_expr(arg1) + self.visit_expr(arg2)
                if isinstance(expr.dtype, PsNumericType) and expr.dtype.is_float():
                    op_counts.float_muls += 1
                elif isinstance(expr.dtype, PsNumericType) and expr.dtype.is_int():
                    op_counts.int_muls += 1
                return op_counts

            case PsDiv(arg1, arg2) | PsIntDiv(arg1, arg2) | PsRem(arg1, arg2):
                if expr.dtype is None:
                    raise PsTypeError(f"Untyped arithmetic expression: {expr}")

                op_counts = self.visit_expr(arg1) + self.visit_expr(arg2)
                if isinstance(expr.dtype, PsNumericType) and expr.dtype.is_float():
                    op_counts.float_divs += 1
                elif isinstance(expr.dtype, PsNumericType) and expr.dtype.is_int():
                    op_counts.int_divs += 1
                return op_counts

            case _:
                return reduce(
                    operator.add,
                    (self.visit_expr(cast(PsExpression, c)) for c in expr.children),
                    OperationCounts(),
                )
