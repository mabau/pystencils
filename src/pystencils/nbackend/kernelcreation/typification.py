from __future__ import annotations

from typing import TypeVar, Any, Sequence, cast

import pymbolic.primitives as pb
from pymbolic.mapper import Mapper

from .context import KernelCreationContext
from ..types import PsAbstractType, PsNumericType
from ..typed_expressions import PsTypedVariable, PsTypedConstant, ExprOrConstant
from ..arrays import PsArrayAccess
from ..ast import PsAstNode, PsExpression, PsAssignment


class TypificationError(Exception):
    """Indicates a fatal error during typification."""


NodeT = TypeVar("NodeT", bound=PsAstNode)


class Typifier(Mapper):
    """Typifier for untyped expressions.

    The typifier, when called with an AST node, will attempt to figure out
    the types for all untyped expressions within the node:

     - Plain variables will be assigned a type according to `ctx.options.default_dtype`.
     - Constants will be converted to typed constants by applying the target type of the current context.
       If the target type is unknown, typification of constants will fail.

    The target type for an expression must either be provided by the user or is inferred from the context.
    The two primary contexts are an assignment, where the target type of the right-hand side expression is
    given by the type of the left-hand side; and the index expression of an array access, where the target
    type is given by `ctx.options.index_dtype`.
    The target type is propagated upward through the expression tree. It is applied to all untyped constants,
    and used to check the correctness of the types of expressions.
    """

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

    def __call__(self, node: NodeT) -> NodeT:
        match node:
            case PsExpression(expr):
                node.expression, _ = self.rec(expr)

            case PsAssignment(lhs, rhs):
                new_lhs, lhs_dtype = self.rec(lhs.expression, None)
                new_rhs, rhs_dtype = self.rec(rhs.expression, lhs_dtype)
                if lhs_dtype != rhs_dtype:
                    #   todo: (optional) automatic cast insertion?
                    raise TypificationError(
                        "Mismatched types in assignment: \n"
                        f"    {lhs} <- {rhs}\n"
                        f"    dtype(lhs) = {lhs_dtype}\n"
                        f"    dtype(rhs) = {rhs_dtype}\n"
                    )
                node.lhs.expression = new_lhs
                node.rhs.expression = new_rhs

            case unknown:
                raise NotImplementedError(f"Don't know how to typify {unknown}")

        return node

    # def rec(self, expr: Any, target_type: PsNumericType | None)

    def typify_expression(
        self, expr: Any, target_type: PsNumericType | None = None
    ) -> ExprOrConstant:
        return self.rec(expr, target_type)

    #   Leaf nodes: Variables, Typed Variables, Constants and TypedConstants

    def map_typed_variable(
        self, var: PsTypedVariable, target_type: PsNumericType | None
    ):
        self._check_target_type(var, var.dtype, target_type)
        return var, var.dtype

    def map_variable(
        self, var: pb.Variable, target_type: PsNumericType | None
    ) -> tuple[PsTypedVariable, PsNumericType]:
        dtype = self._ctx.options.default_dtype
        typed_var = PsTypedVariable(var.name, dtype)
        self._check_target_type(typed_var, dtype, target_type)
        return typed_var, dtype

    def map_constant(
        self, value: Any, target_type: PsNumericType | None
    ) -> tuple[PsTypedConstant, PsNumericType]:
        if isinstance(value, PsTypedConstant):
            self._check_target_type(value, value.dtype, target_type)
            return value, value.dtype
        elif target_type is None:
            raise TypificationError(
                f"Unable to typify constant {value}: Unknown target type in this context."
            )
        else:
            return PsTypedConstant(value, target_type), target_type

    #   Array Access

    def map_array_access(
        self, access: PsArrayAccess, target_type: PsNumericType | None
    ) -> tuple[PsArrayAccess, PsNumericType]:
        self._check_target_type(access, access.dtype, target_type)
        index, _ = self.rec(access.index_tuple[0], self._ctx.options.index_dtype)
        return PsArrayAccess(access.base_ptr, index), cast(PsNumericType, access.dtype)

    #   Arithmetic Expressions

    def _homogenize(
        self,
        expr: pb.Expression,
        args: Sequence[Any],
        target_type: PsNumericType | None,
    ) -> tuple[tuple[ExprOrConstant], PsNumericType]:
        """Typify all arguments of a multi-argument expression with the same type."""
        new_args = [None] * len(args)
        common_type: PsNumericType | None = None

        for i, c in enumerate(args):
            new_args[i], arg_i_type = self.rec(c, target_type)
            if common_type is None:
                common_type = arg_i_type
            elif common_type != arg_i_type:
                raise TypificationError(
                    f"Type mismatch in expression {expr}: Type of operand {i} did not match previous operands\n"
                    f"     Previous type: {common_type}\n"
                    f"  Operand {i} type: {arg_i_type}"
                )

        assert common_type is not None

        return cast(tuple[ExprOrConstant], tuple(new_args)), common_type

    def map_sum(
        self, expr: pb.Sum, target_type: PsNumericType | None
    ) -> tuple[pb.Sum, PsNumericType]:
        new_args, dtype = self._homogenize(expr, expr.children, target_type)
        return pb.Sum(new_args), dtype

    def map_product(
        self, expr: pb.Product, target_type: PsNumericType | None
    ) -> tuple[pb.Product, PsNumericType]:
        new_args, dtype = self._homogenize(expr, expr.children, target_type)
        return pb.Product(new_args), dtype

    def _check_target_type(
        self,
        expr: ExprOrConstant,
        expr_type: PsAbstractType,
        target_type: PsNumericType | None,
    ):
        if target_type is not None and expr_type != target_type:
            raise TypificationError(
                f"Type mismatch at expression {expr}: Expression type did not match the context's target type\n"
                f"  Expression type: {expr_type}\n"
                f"      Target type: {target_type}"
            )
