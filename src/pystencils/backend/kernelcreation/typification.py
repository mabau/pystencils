from __future__ import annotations

from typing import TypeVar, Any

import pymbolic.primitives as pb
from pymbolic.mapper import Mapper

from .context import KernelCreationContext
from ..types import PsAbstractType, PsNumericType, PsStructType, deconstify
from ..typed_expressions import PsTypedVariable, PsTypedConstant, ExprOrConstant
from ..arrays import PsArrayAccess, PsVectorArrayAccess
from ..ast import PsAstNode, PsBlock, PsExpression, PsAssignment
from ..functions import PsMathFunction

__all__ = ["Typifier"]


class TypificationError(Exception):
    """Indicates a fatal error during typification."""


NodeT = TypeVar("NodeT", bound=PsAstNode)


class DeferredTypedConstant(PsTypedConstant):
    """Special subclass for constants whose types cannot be determined yet at the time of their creation.

    Outside of the typifier, a DeferredTypedConstant acts exactly the same way as a PsTypedConstant.
    """

    def __init__(self, value: Any):
        self._value_deferred = value

    def resolve(self, dtype: PsNumericType):
        super().__init__(self._value_deferred, dtype)


class TypeContext:
    def __init__(self, target_type: PsAbstractType | None):
        self._target_type = deconstify(target_type) if target_type is not None else None
        self._deferred_constants: list[DeferredTypedConstant] = []

    def make_constant(self, value: Any) -> PsTypedConstant:
        if self._target_type is None:
            dc = DeferredTypedConstant(value)
            self._deferred_constants.append(dc)
            return dc
        elif not isinstance(self._target_type, PsNumericType):
            raise TypificationError(
                f"Can't typify constant with non-numeric type {self._target_type}"
            )
        else:
            return PsTypedConstant(value, self._target_type)

    def apply(self, target_type: PsAbstractType):
        assert self._target_type is None, "Type context was already resolved"
        self._target_type = deconstify(target_type)

        for dc in self._deferred_constants:
            if not isinstance(self._target_type, PsNumericType):
                raise TypificationError(
                    f"Can't typify constant with non-numeric type {self._target_type}"
                )
            dc.resolve(self._target_type)

        self._deferred_constants = []

    @property
    def target_type(self) -> PsAbstractType | None:
        return self._target_type


class Typifier(Mapper):
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
        match node:
            case PsBlock([*statements]):
                node.statements = [self(s) for s in statements]

            case PsExpression(expr):
                node.expression = self.rec(expr, TypeContext(None))

            case PsAssignment(lhs, rhs):
                tc = TypeContext(None)
                #   LHS defines target type; type context carries it to RHS
                new_lhs = self.rec(lhs.expression, tc)
                assert tc.target_type is not None
                new_rhs = self.rec(rhs.expression, tc)

                node.lhs.expression = new_lhs
                node.rhs.expression = new_rhs

            case unknown:
                raise NotImplementedError(f"Don't know how to typify {unknown}")

        return node

    """
    def rec(self, expr: Any, tc: TypeContext) -> ExprOrConstant

    All visitor methods take an expression and the current type context.
    They shall return the typified expression, or throw `TypificationError` if typification fails.
    """

    def typify_expression(
        self, expr: Any, target_type: PsNumericType | None = None
    ) -> ExprOrConstant:
        tc = TypeContext(target_type)
        return self.rec(expr, tc)

    #   Leaf nodes: Variables, Typed Variables, Constants and TypedConstants

    def map_typed_variable(self, var: PsTypedVariable, tc: TypeContext):
        self._apply_target_type(var, var.dtype, tc)
        return var

    def map_variable(self, var: pb.Variable, tc: TypeContext) -> PsTypedVariable:
        dtype = self._ctx.default_dtype
        typed_var = PsTypedVariable(var.name, dtype)
        self._apply_target_type(typed_var, dtype, tc)
        return typed_var

    def map_constant(self, value: Any, tc: TypeContext) -> PsTypedConstant:
        if isinstance(value, PsTypedConstant):
            self._apply_target_type(value, value.dtype, tc)
            return value

        return tc.make_constant(value)

    #   Array Accesses and Lookups

    def map_array_access(self, access: PsArrayAccess, tc: TypeContext) -> PsArrayAccess:
        self._apply_target_type(access, access.dtype, tc)
        index = self.rec(access.index_tuple[0], TypeContext(self._ctx.index_dtype))
        return PsArrayAccess(access.base_ptr, index)

    def map_vector_array_access(
        self, access: PsVectorArrayAccess, tc: TypeContext
    ) -> PsVectorArrayAccess:
        self._apply_target_type(access, access.dtype, tc)
        base_index = self.rec(access.base_index, TypeContext(self._ctx.index_dtype))
        return PsVectorArrayAccess(
            access.base_ptr, base_index, access.dtype.vector_entries, access.stride
        )

    def map_lookup(self, lookup: pb.Lookup, tc: TypeContext) -> pb.Lookup:
        aggr_tc = TypeContext(None)
        aggregate = self.rec(lookup.aggregate, aggr_tc)
        aggr_type = aggr_tc.target_type

        if not isinstance(aggr_type, PsStructType):
            raise TypificationError("Aggregate type of lookup was not a struct type.")

        member = aggr_type.get_member(lookup.name)
        if member is None:
            raise TypificationError(
                f"Aggregate of type {aggr_type} does not have a member {member}."
            )

        self._apply_target_type(lookup, member.dtype, tc)
        return pb.Lookup(aggregate, member.name)

    #   Arithmetic Expressions

    def map_sum(self, expr: pb.Sum, tc: TypeContext) -> pb.Sum:
        return pb.Sum(tuple(self.rec(c, tc) for c in expr.children))

    def map_product(self, expr: pb.Product, tc: TypeContext) -> pb.Product:
        return pb.Product(tuple(self.rec(c, tc) for c in expr.children))

    def map_quotient(self, expr: pb.Quotient, tc: TypeContext) -> pb.Quotient:
        return pb.Quotient(self.rec(expr.num, tc), self.rec(expr.den, tc))

    #   Functions

    def map_call(self, expr: pb.Call, tc: TypeContext) -> pb.Call:
        func = expr.function
        args = expr.parameters
        match func:
            case PsMathFunction():
                return pb.Call(func, tuple(self.rec(arg, tc) for arg in args))
            case _:
                raise TypificationError(f"Don't know how to typify calls to {func}")

    #   Internals

    def _apply_target_type(
        self, expr: ExprOrConstant, expr_type: PsAbstractType, tc: TypeContext
    ):
        if tc.target_type is None:
            tc.apply(expr_type)
        elif deconstify(expr_type) != tc.target_type:
            raise TypificationError(
                f"Type mismatch at expression {expr}: Expression type did not match the context's target type\n"
                f"  Expression type: {expr_type}\n"
                f"      Target type: {tc.target_type}"
            )
