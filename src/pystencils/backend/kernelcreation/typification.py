from __future__ import annotations

from typing import TypeVar

from .context import KernelCreationContext
from ...types import PsType, PsNumericType, PsStructType, PsIntegerType, deconstify
from ..ast.structural import PsAstNode, PsBlock, PsLoop, PsExpression, PsAssignment
from ..ast.expressions import (
    PsSymbolExpr,
    PsConstantExpr,
    PsBinOp,
    PsArrayAccess,
    PsLookup,
    PsCall,
)
from ..functions import PsMathFunction

__all__ = ["Typifier"]


class TypificationError(Exception):
    """Indicates a fatal error during typification."""


NodeT = TypeVar("NodeT", bound=PsAstNode)


class TypeContext:
    def __init__(self, target_type: PsType | None = None):
        self._target_type = deconstify(target_type) if target_type is not None else None
        self._deferred_constants: list[PsConstantExpr] = []

    def typify_constant(self, constexpr: PsConstantExpr) -> None:
        if self._target_type is None:
            self._deferred_constants.append(constexpr)
        elif not isinstance(self._target_type, PsNumericType):
            raise TypificationError(
                f"Can't typify constant with non-numeric type {self._target_type}"
            )
        else:
            constexpr.constant.apply_dtype(self._target_type)

    def apply_and_check(self, expr: PsExpression, expr_type: PsType):
        """
        If no target type has been set yet, establishes expr_type as the target type
        and typifies all deferred expressions.

        Otherwise, checks if expression type and target type are compatible.
        """
        expr_type = deconstify(expr_type)

        if self._target_type is None:
            self._target_type = expr_type

            for dc in self._deferred_constants:
                if not isinstance(self._target_type, PsNumericType):
                    raise TypificationError(
                        f"Can't typify constant with non-numeric type {self._target_type}"
                    )
                dc.constant.apply_dtype(self._target_type)

            self._deferred_constants = []

        elif expr_type != self._target_type:
            raise TypificationError(
                f"Type mismatch at expression {expr}: Expression type did not match the context's target type\n"
                f"  Expression type: {expr_type}\n"
                f"      Target type: {self._target_type}"
            )

    @property
    def target_type(self) -> PsType | None:
        return self._target_type


class Typifier:
    """Typifier for untyped expressions.

    The typifier, when called with an AST node, will attempt to figure out
    the types for all untyped expressions within the node.
    Plain variables will be assigned a type according to `ctx.options.default_dtype`,
    constants will be converted to typed constants according to the contextual typing scheme
    described below.

    Contextual Typing
    -----------------

    The contextual typifier covers the expression tree with disjoint typing contexts.
    The idea is that all nodes covered by a typing context must have the exact same type.
    Starting at an expression's root, the typifier attempts to expand a typing context as far as possible
    toward the leaves.
    This happens implicitly during the recursive traversal of the expression tree.

    At an interior node, which is modelled as a function applied to a number of arguments, producing a result,
    that function's signature governs context expansion. Let T be the function's return type; then the context
    is expanded to each argument expression that also is of type T.

    If a function parameter is of type S != T, a new type context is created for it. If the type S is already fixed
    by the function signature, it will be the target type of the new context.

    At the tree's leaves, types are applied and checked. By the above propagation rule, all leaves that share a typing
    context must have the exact same type (modulo constness). There the actual type checking happens.
    If a variable is encountered and the context does not yet have a target type, it is set to the variable's type.
    If a constant is encountered, it is typified using the current target type.
    If no target type is known yet, the constant will first be instantiated as a DeferredTypedConstant,
    and stashed in the context.
    As soon as the context learns its target type, it is applied to all deferred constants.

    In addition to leaves, some interior nodes may also have to be checked against the target type.
    In particular, these are array accesses, struct member accesses, and calls to functions with a fixed
    return type.

    When a context is 'closed' during the recursive unwinding, it shall be an error if it still contains unresolved
    constants.

    TODO: The context shall keep track of it's target type's origin to aid in producing helpful error messages.
    """

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

    def __call__(self, node: NodeT) -> NodeT:
        if isinstance(node, PsExpression):
            self.visit_expr(node, TypeContext())
        else:
            self.visit(node)
        return node

    def typify_expression(
        self, expr: PsExpression, target_type: PsNumericType | None = None
    ) -> PsExpression:
        tc = TypeContext(target_type)
        self.visit_expr(expr, tc)
        return expr

    def visit(self, node: PsAstNode) -> None:
        """Recursive processing of structural nodes"""
        match node:
            case PsBlock([*statements]):
                for s in statements:
                    self.visit(s)

            case PsAssignment(lhs, rhs):
                tc = TypeContext()
                #   LHS defines target type; type context carries it to RHS
                self.visit_expr(lhs, tc)
                assert tc.target_type is not None
                self.visit_expr(rhs, tc)

            case PsLoop(ctr, start, stop, step, body):
                if ctr.symbol.dtype is None:
                    ctr.symbol.apply_dtype(self._ctx.index_dtype)

                tc = TypeContext(ctr.symbol.dtype)
                self.visit_expr(start, tc)
                self.visit_expr(stop, tc)
                self.visit_expr(step, tc)

                self.visit(body)

            case _:
                raise NotImplementedError(f"Can't typify {node}")

    def visit_expr(self, expr: PsExpression, tc: TypeContext) -> None:
        """Recursive processing of expression nodes"""
        match expr:
            case PsSymbolExpr(symb):
                if symb.dtype is None:
                    dtype = self._ctx.default_dtype
                    symb.apply_dtype(dtype)
                tc.apply_and_check(expr, symb.get_dtype())

            case PsConstantExpr(constant):
                if constant.dtype is not None:
                    tc.apply_and_check(expr, constant.get_dtype())
                else:
                    tc.typify_constant(expr)

            case PsArrayAccess(_, idx):
                tc.apply_and_check(expr, expr.dtype)
                
                index_tc = TypeContext()
                self.visit_expr(idx, index_tc)
                if index_tc.target_type is None:
                    index_tc.apply_and_check(idx, self._ctx.index_dtype)
                elif not isinstance(index_tc.target_type, PsIntegerType):
                    raise TypificationError(
                        f"Array index is not of integer type: {idx} has type {index_tc.target_type}"
                    )

            case PsLookup(aggr, member_name):
                aggr_tc = TypeContext(None)
                self.visit_expr(aggr, aggr_tc)
                aggr_type = aggr_tc.target_type

                if not isinstance(aggr_type, PsStructType):
                    raise TypificationError(
                        "Aggregate type of lookup was not a struct type."
                    )

                member = aggr_type.find_member(member_name)
                if member is None:
                    raise TypificationError(
                        f"Aggregate of type {aggr_type} does not have a member {member}."
                    )

                tc.apply_and_check(expr, member.dtype)

            case PsBinOp(op1, op2):
                self.visit_expr(op1, tc)
                self.visit_expr(op2, tc)

            case PsCall(function, args):
                match function:
                    case PsMathFunction():
                        for arg in args:
                            self.visit_expr(arg, tc)
                    case _:
                        raise TypificationError(
                            f"Don't know how to typify calls to {function}"
                        )

            case _:
                raise NotImplementedError(f"Can't typify {expr}")
