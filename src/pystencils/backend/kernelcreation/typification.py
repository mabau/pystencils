from __future__ import annotations

from typing import TypeVar

from .context import KernelCreationContext
from ...types import (
    PsType,
    PsNumericType,
    PsStructType,
    PsIntegerType,
    PsArrayType,
    PsDereferencableType,
    PsPointerType,
    PsBoolType,
    deconstify,
)
from ..ast.structural import (
    PsAstNode,
    PsBlock,
    PsLoop,
    PsConditional,
    PsExpression,
    PsAssignment,
)
from ..ast.expressions import (
    PsArrayAccess,
    PsArrayInitList,
    PsBinOp,
    PsBitwiseAnd,
    PsBitwiseOr,
    PsBitwiseXor,
    PsCall,
    PsCast,
    PsDeref,
    PsAddressOf,
    PsConstantExpr,
    PsIntDiv,
    PsLeftShift,
    PsLookup,
    PsRightShift,
    PsSubscript,
    PsSymbolExpr,
)
from ..functions import PsMathFunction

__all__ = ["Typifier"]


class TypificationError(Exception):
    """Indicates a fatal error during typification."""


NodeT = TypeVar("NodeT", bound=PsAstNode)


class TypeContext:
    def __init__(self, target_type: PsType | None = None):
        self._target_type = deconstify(target_type) if target_type is not None else None
        self._deferred_exprs: list[PsExpression] = []

    def apply_dtype(self, expr: PsExpression | None, dtype: PsType):
        """Applies the given ``dtype`` to the given expression inside this type context.

        The given expression will be covered by this type context.
        If the context's target_type is already known, it must be compatible with the given dtype.
        If the target type is still unknown, target_type is set to dtype and retroactively applied
        to all deferred expressions.
        """

        dtype = deconstify(dtype)

        if self._target_type is not None and dtype != self._target_type:
            raise TypificationError(
                f"Type mismatch at expression {expr}: Expression type did not match the context's target type\n"
                f"  Expression type: {dtype}\n"
                f"      Target type: {self._target_type}"
            )
        else:
            self._target_type = dtype
            self._propagate_target_type()

        if expr is not None:
            if expr.dtype is None:
                self._apply_target_type(expr)
            elif deconstify(expr.dtype) != self._target_type:
                raise TypificationError(
                    "Type conflict: Predefined expression type did not match the context's target type\n"
                    f"  Expression type: {dtype}\n"
                    f"      Target type: {self._target_type}"
                )

    def infer_dtype(self, expr: PsExpression):
        """Infer the data type for the given expression.

        If the target_type of this context is already known, it will be applied to the given expression.
        Otherwise, the expression is deferred, and a type will be applied to it as soon as `apply_type` is
        called on this context.

        It the expression already has a data type set, it must be equal to the inferred type.
        """

        if self._target_type is None:
            self._deferred_exprs.append(expr)
        else:
            self._apply_target_type(expr)

    def _propagate_target_type(self):
        for expr in self._deferred_exprs:
            self._apply_target_type(expr)
        self._deferred_exprs = []

    def _apply_target_type(self, expr: PsExpression):
        assert self._target_type is not None

        if expr.dtype is not None:
            if deconstify(expr.dtype) != self.target_type:
                raise TypificationError(
                    f"Type mismatch at expression {expr}: Expression type did not match the context's target type\n"
                    f"  Expression type: {expr.dtype}\n"
                    f"      Target type: {self._target_type}"
                )
        else:
            match expr:
                case PsConstantExpr(c):
                    if not isinstance(self._target_type, PsNumericType):
                        raise TypificationError(
                            f"Can't typify constant with non-numeric type {self._target_type}"
                        )
                    c.apply_dtype(self._target_type)
                
                case PsSymbolExpr(symb):
                    symb.apply_dtype(self._target_type)

                case (
                    PsIntDiv()
                    | PsLeftShift()
                    | PsRightShift()
                    | PsBitwiseAnd()
                    | PsBitwiseXor()
                    | PsBitwiseOr()
                ) if not isinstance(self._target_type, PsIntegerType):
                    raise TypificationError(
                        f"Integer operation encountered in non-integer type context:\n"
                        f"    Expression: {expr}"
                        f"  Type Context: {self._target_type}"
                    )

            expr.dtype = self._target_type
        # endif

    @property
    def target_type(self) -> PsType | None:
        return self._target_type


class Typifier:
    """Apply data types to expressions.

    The Typifier will traverse the AST and apply a contextual typing scheme to figure out
    the data types of all encountered expressions.
    To this end, it covers each expression tree with a set of disjoint typing contexts.
    All nodes covered by the same typing context must have the same type.

    Starting from an expression's root, a typing context is implicitly expanded through
    the recursive descent into a node's children. In particular, a child is typified within
    the same context as its parent if the node's semantics require parent and child to have
    the same type (e.g. at arithmetic operators, mathematical functions, etc.).
    If a node's child is required to have a different type, a new context is opened.

    For each typing context, its target type is prescribed by the first node encountered during traversal
    whose type is fixed according to its typing rules. All other nodes covered by the context must share
    that type.

    The types of arithmetic operators, mathematical functions, and untyped constants are
    inferred from their context's target type. If one of these is encountered while no target type is set yet
    in the context, the expression is deferred by storing it in the context, and will be assigned a type as soon
    as the target type is fixed.

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
        self, expr: PsExpression, target_type: PsType | None = None
    ) -> tuple[PsExpression, PsType]:
        tc = TypeContext(target_type)
        self.visit_expr(expr, tc)

        if tc.target_type is None:
            raise TypificationError(f"Unable to determine type for {expr}")

        return expr, tc.target_type

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

            case PsConditional(cond, branch_true, branch_false):
                cond_tc = TypeContext(PsBoolType(const=True))
                self.visit_expr(cond, cond_tc)

                self.visit(branch_true)

                if branch_false is not None:
                    self.visit(branch_false)

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
            case PsSymbolExpr(_):
                if expr.dtype is None:
                    tc.apply_dtype(expr, self._ctx.default_dtype)
                else:
                    tc.apply_dtype(expr, expr.dtype)

            case PsConstantExpr(_):
                tc.infer_dtype(expr)

            case PsArrayAccess(bptr, idx):
                tc.apply_dtype(expr, bptr.array.element_type)

                index_tc = TypeContext()
                self.visit_expr(idx, index_tc)
                if index_tc.target_type is None:
                    index_tc.apply_dtype(idx, self._ctx.index_dtype)
                elif not isinstance(index_tc.target_type, PsIntegerType):
                    raise TypificationError(
                        f"Array index is not of integer type: {idx} has type {index_tc.target_type}"
                    )

            case PsSubscript(arr, idx):
                arr_tc = TypeContext()
                self.visit_expr(arr, arr_tc)

                if not isinstance(arr_tc.target_type, PsDereferencableType):
                    raise TypificationError(
                        "Type of subscript base is not subscriptable."
                    )

                tc.apply_dtype(expr, arr_tc.target_type.base_type)

                index_tc = TypeContext()
                self.visit_expr(idx, index_tc)
                if index_tc.target_type is None:
                    index_tc.apply_dtype(idx, self._ctx.index_dtype)
                elif not isinstance(index_tc.target_type, PsIntegerType):
                    raise TypificationError(
                        f"Subscript index is not of integer type: {idx} has type {index_tc.target_type}"
                    )

            case PsDeref(ptr):
                ptr_tc = TypeContext()
                self.visit_expr(ptr, ptr_tc)

                if not isinstance(ptr_tc.target_type, PsDereferencableType):
                    raise TypificationError(
                        "Type of argument to a Deref is not dereferencable"
                    )

                tc.apply_dtype(expr, ptr_tc.target_type.base_type)

            case PsAddressOf(arg):
                arg_tc = TypeContext()
                self.visit_expr(arg, arg_tc)

                if arg_tc.target_type is None:
                    raise TypificationError(
                        f"Unable to determine type of argument to AddressOf: {arg}"
                    )

                ptr_type = PsPointerType(arg_tc.target_type, True)
                tc.apply_dtype(expr, ptr_type)

            case PsLookup(aggr, member_name):
                aggr_tc = TypeContext(None)
                self.visit_expr(aggr, aggr_tc)
                aggr_type = aggr_tc.target_type

                if not isinstance(aggr_type, PsStructType):
                    raise TypificationError(
                        "Aggregate type of lookup is not a struct type."
                    )

                member = aggr_type.find_member(member_name)
                if member is None:
                    raise TypificationError(
                        f"Aggregate of type {aggr_type} does not have a member {member}."
                    )

                tc.apply_dtype(expr, member.dtype)

            case PsBinOp(op1, op2):
                self.visit_expr(op1, tc)
                self.visit_expr(op2, tc)
                tc.infer_dtype(expr)

            case PsCall(function, args):
                match function:
                    case PsMathFunction():
                        for arg in args:
                            self.visit_expr(arg, tc)
                        tc.infer_dtype(expr)
                    case _:
                        raise TypificationError(
                            f"Don't know how to typify calls to {function}"
                        )

            case PsArrayInitList(items):
                items_tc = TypeContext()
                for item in items:
                    self.visit_expr(item, items_tc)

                if items_tc.target_type is None:
                    if tc.target_type is None:
                        raise TypificationError(f"Unable to infer type of array {expr}")
                    elif not isinstance(tc.target_type, PsArrayType):
                        raise TypificationError(
                            f"Cannot apply type {tc.target_type} to an array initializer."
                        )
                    elif (
                        tc.target_type.length is not None
                        and tc.target_type.length != len(items)
                    ):
                        raise TypificationError(
                            "Array size mismatch: Cannot typify initializer list with "
                            f"{len(items)} items as {tc.target_type}"
                        )
                    else:
                        items_tc.apply_dtype(None, tc.target_type.base_type)
                else:
                    arr_type = PsArrayType(items_tc.target_type, len(items))
                    tc.apply_dtype(expr, arr_type)

            case PsCast(dtype, arg):
                self.visit_expr(arg, TypeContext())
                tc.apply_dtype(expr, dtype)

            case _:
                raise NotImplementedError(f"Can't typify {expr}")
