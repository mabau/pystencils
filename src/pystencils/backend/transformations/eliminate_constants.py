from typing import cast

from ..kernelcreation.context import KernelCreationContext

from ..ast import PsAstNode
from ..ast.expressions import (
    PsExpression,
    PsConstantExpr,
    PsSymbolExpr,
    PsBinOp,
    PsAdd,
    PsSub,
    PsMul,
    PsDiv,
)

from ..constants import PsConstant
from ...types import PsIntegerType, PsIeeeFloatType


__all__ = ["EliminateConstants"]


class ECContext:
    def __init__(self):
        pass


class EliminateConstants:
    """Eliminate constant expressions in various ways.

    - Constant folding: Nontrivial constant integer (and optionally floating point) expressions
      are evaluated and replaced by their result
    - Idempotence elimination: Idempotent operations (e.g. addition of zero, multiplication with one)
      are replaced by their result
    - Dominance elimination: Multiplication by zero is replaced by zero
    - Constant extraction: Optionally, nontrivial constant expressions are extracted and listed at the beginning of
      the outermost block.
    """

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx
        
        self._fold_integers = True
        self._fold_floats = False
        self._extract_constant_exprs = True

    def __call__(self, node: PsAstNode) -> PsAstNode:
        return self.visit(node)

    def visit(self, node: PsAstNode) -> PsAstNode:
        match node:
            case PsExpression():
                transformed_expr, _ = self.visit_expr(node)
                return transformed_expr
            case _:
                node.children = [self.visit(c) for c in node.children]
                return node

    def visit_expr(self, expr: PsExpression) -> tuple[PsExpression, bool]:
        """Transformation of expressions.

        Returns:
            (transformed_expr, is_const): The tranformed expression, and a flag indicating whether it is constant
        """
        #   Return constants as they are
        if isinstance(expr, PsConstantExpr):
            return expr, True
        
        #   Shortcut symbols
        if isinstance(expr, PsSymbolExpr):
            return expr, False

        subtree_results = [
            self.visit_expr(cast(PsExpression, c)) for c in expr.children
        ]
        expr.children = [r[0] for r in subtree_results]
        subtree_constness = [r[1] for r in subtree_results]

        #   Eliminate idempotence and dominance
        match expr:
            #   Additive idempotence: Addition and subtraction of zero
            case PsAdd(PsConstantExpr(c), other_op) if c.value == 0:
                return other_op, all(subtree_constness)

            case PsAdd(other_op, PsConstantExpr(c)) if c.value == 0:
                return other_op, all(subtree_constness)

            case PsSub(other_op, PsConstantExpr(c)) if c.value == 0:
                return other_op, all(subtree_constness)

            #   Additive idempotence: Subtraction from zero
            case PsSub(PsConstantExpr(c), other_op) if c.value == 0:
                other_transformed, is_const = self.visit_expr(-other_op)
                return other_transformed, is_const

            #   Multiplicative idempotence: Multiplication with and division by one
            case PsMul(PsConstantExpr(c), other_op) if c.value == 1:
                return other_op, all(subtree_constness)

            case PsMul(other_op, PsConstantExpr(c)) if c.value == 1:
                return other_op, all(subtree_constness)

            case PsDiv(other_op, PsConstantExpr(c)) if c.value == 1:
                return other_op, all(subtree_constness)

            #   Multiplicative dominance: 0 * x = 0
            case PsMul(PsConstantExpr(c), other_op) if c.value == 0:
                return PsConstantExpr(c), True

            case PsMul(other_op, PsConstantExpr(c)) if c.value == 0:
                return PsConstantExpr(c), True

        # end match: no idempotence or dominance encountered

        #   Detect constant expressions
        if all(subtree_constness):
            #   Fold binary expressions where possible
            if isinstance(expr, PsBinOp):
                op1_transformed = expr.operand1
                op2_transformed = expr.operand2

                if isinstance(op1_transformed, PsConstantExpr) and isinstance(
                    op2_transformed, PsConstantExpr
                ):
                    v1 = op1_transformed.constant.value
                    v2 = op2_transformed.constant.value

                    # assume they are of equal type
                    dtype = op1_transformed.constant.dtype

                    is_int = isinstance(dtype, PsIntegerType)
                    is_float = isinstance(dtype, PsIeeeFloatType)

                    if (self._fold_integers and is_int) or (
                        self._fold_floats and is_float
                    ):
                        py_operator = expr.python_operator

                        folded = None
                        if py_operator is not None:
                            folded = PsConstant(
                                py_operator(v1, v2),
                                dtype,
                            )
                        elif isinstance(expr, PsDiv):
                            if isinstance(dtype, PsIntegerType):
                                folded = PsConstant(v1 // v2, dtype)
                            elif isinstance(dtype, PsIeeeFloatType):
                                folded = PsConstant(v1 / v2, dtype)

                        if folded is not None:
                            return PsConstantExpr(folded), True

                expr.operand1 = op1_transformed
                expr.operand2 = op2_transformed
                return expr, True
        # end if: no constant expressions encountered

        #   Any other expressions are not considered constant even if their arguments are
        return expr, False
