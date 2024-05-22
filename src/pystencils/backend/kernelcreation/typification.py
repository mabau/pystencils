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
    constify,
)
from ..ast.structural import (
    PsAstNode,
    PsBlock,
    PsLoop,
    PsConditional,
    PsExpression,
    PsAssignment,
    PsDeclaration,
    PsComment,
)
from ..ast.expressions import (
    PsArrayAccess,
    PsArrayInitList,
    PsBinOp,
    PsIntOpTrait,
    PsNumericOpTrait,
    PsBoolOpTrait,
    PsCall,
    PsCast,
    PsDeref,
    PsAddressOf,
    PsConstantExpr,
    PsLookup,
    PsSubscript,
    PsSymbolExpr,
    PsLiteralExpr,
    PsRel,
    PsNeg,
    PsNot,
)
from ..functions import PsMathFunction, CFunction

__all__ = ["Typifier"]


class TypificationError(Exception):
    """Indicates a fatal error during typification."""


NodeT = TypeVar("NodeT", bound=PsAstNode)


class TypeContext:
    """Typing context, with support for type inference and checking.

    Instances of this class are used to propagate and check data types across expression subtrees
    of the AST. Each type context has:

     - A target type `target_type`, which shall be applied to all expressions it covers
     - A set of restrictions on the target type:
       - `require_nonconst` to make sure the target type is not `const`, as required on assignment left-hand sides
       - Additional restrictions may be added in the future.
    """

    def __init__(
        self, target_type: PsType | None = None, require_nonconst: bool = False
    ):
        self._require_nonconst = require_nonconst
        self._deferred_exprs: list[PsExpression] = []

        self._target_type = (
            self._fix_constness(target_type) if target_type is not None else None
        )

    @property
    def target_type(self) -> PsType | None:
        return self._target_type

    @property
    def require_nonconst(self) -> bool:
        return self._require_nonconst

    def apply_dtype(self, dtype: PsType, expr: PsExpression | None = None):
        """Applies the given ``dtype`` to this type context, and optionally to the given expression.

        If the context's target_type is already known, it must be compatible with the given dtype.
        If the target type is still unknown, target_type is set to dtype and retroactively applied
        to all deferred expressions.

        If an expression is specified, it will be covered by the type context.
        If the expression already has a data type set, it must be compatible with the target type
        and will be replaced by it.
        """

        dtype = self._fix_constness(dtype)

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
            self._apply_target_type(expr)

    def infer_dtype(self, expr: PsExpression):
        """Infer the data type for the given expression.

        If the target_type of this context is already known, it will be applied to the given expression.
        Otherwise, the expression is deferred, and a type will be applied to it as soon as `apply_type` is
        called on this context.

        If the expression already has a data type set, it must be compatible with the target type
        and will be replaced by it.
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
            if not self._compatible(expr.dtype):
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
                    if c.dtype is None:
                        expr.constant = c.interpret_as(self._target_type)
                    elif not self._compatible(c.dtype):
                        raise TypificationError(
                            f"Type mismatch at constant {c}: Constant type did not match the context's target type\n"
                            f"  Constant type: {c.dtype}\n"
                            f"    Target type: {self._target_type}"
                        )
                
                case PsLiteralExpr(lit):
                    if not self._compatible(lit.dtype):
                        raise TypificationError(
                            f"Type mismatch at literal {lit}: Literal type did not match the context's target type\n"
                            f"   Literal type: {lit.dtype}\n"
                            f"    Target type: {self._target_type}"
                        )

                case PsSymbolExpr(symb):
                    assert symb.dtype is not None
                    if not self._compatible(symb.dtype):
                        raise TypificationError(
                            f"Type mismatch at symbol {symb}: Symbol type did not match the context's target type\n"
                            f"    Symbol type: {symb.dtype}\n"
                            f"    Target type: {self._target_type}"
                        )

                case PsNumericOpTrait() if not isinstance(
                    self._target_type, PsNumericType
                ) or isinstance(self._target_type, PsBoolType):
                    #   FIXME: PsBoolType derives from PsNumericType, but is not numeric
                    raise TypificationError(
                        f"Numerical operation encountered in non-numerical type context:\n"
                        f"    Expression: {expr}"
                        f"  Type Context: {self._target_type}"
                    )

                case PsIntOpTrait() if not isinstance(self._target_type, PsIntegerType):
                    raise TypificationError(
                        f"Integer operation encountered in non-integer type context:\n"
                        f"    Expression: {expr}"
                        f"  Type Context: {self._target_type}"
                    )

                case PsBoolOpTrait() if not isinstance(self._target_type, PsBoolType):
                    raise TypificationError(
                        f"Boolean operation encountered in non-boolean type context:\n"
                        f"    Expression: {expr}"
                        f"  Type Context: {self._target_type}"
                    )
        # endif
        expr.dtype = self._target_type

    def _compatible(self, dtype: PsType):
        """Checks whether the given data type is compatible with the context's target type.

        If the target type is ``const``, they must be equal up to const qualification;
        if the target type is not ``const``, `dtype` must match it exactly.
        """
        assert self._target_type is not None
        if self._target_type.const:
            return constify(dtype) == self._target_type
        else:
            return dtype == self._target_type

    def _fix_constness(self, dtype: PsType, expr: PsExpression | None = None):
        if self._require_nonconst:
            if dtype.const:
                if expr is None:
                    raise TypificationError(
                        f"Type mismatch: Encountered {dtype} in non-constant context."
                    )
                else:
                    raise TypificationError(
                        f"Type mismatch at expression {expr}: Encountered {dtype} in non-constant context."
                    )
            return dtype
        else:
            return constify(dtype)


class Typifier:
    """Apply data types to expressions.

    **Contextual Typing**

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

    **Typing Rules**

    The following general rules apply:

     - The context's `default_dtype` is applied to all untyped symbols
     - By default, all expressions receive a ``const`` type unless they occur on a (non-declaration) assignment's
       left-hand side

    **Typing of symbol expressions**

    Some expressions (`PsSymbolExpr`, `PsArrayAccess`) encapsulate symbols and inherit their data types, but
    not necessarily their const-qualification.
    A symbol with non-``const`` type may occur in a `PsSymbolExpr` with ``const`` type,
    and an array base pointer with non-``const`` base type may be nested in a ``const`` `PsArrayAccess`,
    but not vice versa.
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

            case PsDeclaration(lhs, rhs):
                tc = TypeContext()
                #   LHS defines target type; type context carries it to RHS
                self.visit_expr(lhs, tc)
                assert tc.target_type is not None
                self.visit_expr(rhs, tc)

            case PsAssignment(lhs, rhs):
                tc_lhs = TypeContext(require_nonconst=True)
                self.visit_expr(lhs, tc_lhs)
                assert tc_lhs.target_type is not None

                tc_rhs = TypeContext(tc_lhs.target_type, require_nonconst=False)
                self.visit_expr(rhs, tc_rhs)

            case PsConditional(cond, branch_true, branch_false):
                cond_tc = TypeContext(PsBoolType())
                self.visit_expr(cond, cond_tc)

                self.visit(branch_true)

                if branch_false is not None:
                    self.visit(branch_false)

            case PsLoop(ctr, start, stop, step, body):
                if ctr.symbol.dtype is None:
                    ctr.symbol.apply_dtype(self._ctx.index_dtype)

                tc_index = TypeContext(ctr.symbol.dtype)
                self.visit_expr(start, tc_index)
                self.visit_expr(stop, tc_index)
                self.visit_expr(step, tc_index)

                self.visit(body)

            case PsComment():
                pass

            case _:
                raise NotImplementedError(f"Can't typify {node}")

    def visit_expr(self, expr: PsExpression, tc: TypeContext) -> None:
        """Recursive processing of expression nodes.

        This method opens, expands, and closes typing contexts according to the respective expression's
        typing rules. It may add or check restrictions only when opening or closing a type context.

        The actual type inference and checking during context expansion are performed by the methods
        of `TypeContext`. ``visit_expr`` tells the typing context how to handle an expression by calling
        either ``apply_dtype`` or ``infer_dtype``.
        """
        match expr:
            case PsSymbolExpr(_):
                if expr.symbol.dtype is None:
                    expr.symbol.dtype = self._ctx.default_dtype

                tc.apply_dtype(expr.symbol.dtype, expr)

            case PsConstantExpr(c):
                if c.dtype is not None:
                    tc.apply_dtype(c.dtype, expr)
                else:
                    tc.infer_dtype(expr)

            case PsLiteralExpr(lit):
                tc.apply_dtype(lit.dtype, expr)

            case PsArrayAccess(bptr, idx):
                tc.apply_dtype(bptr.array.element_type, expr)

                index_tc = TypeContext()
                self.visit_expr(idx, index_tc)
                if index_tc.target_type is None:
                    index_tc.apply_dtype(self._ctx.index_dtype, idx)
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

                tc.apply_dtype(arr_tc.target_type.base_type, expr)

                index_tc = TypeContext()
                self.visit_expr(idx, index_tc)
                if index_tc.target_type is None:
                    index_tc.apply_dtype(self._ctx.index_dtype, idx)
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

                tc.apply_dtype(ptr_tc.target_type.base_type, expr)

            case PsAddressOf(arg):
                arg_tc = TypeContext()
                self.visit_expr(arg, arg_tc)

                if arg_tc.target_type is None:
                    raise TypificationError(
                        f"Unable to determine type of argument to AddressOf: {arg}"
                    )

                ptr_type = PsPointerType(arg_tc.target_type, const=True)
                tc.apply_dtype(ptr_type, expr)

            case PsLookup(aggr, member_name):
                #   Members of a struct type inherit the struct type's `const` qualifier
                aggr_tc = TypeContext(None, require_nonconst=tc.require_nonconst)
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

                member_type = member.dtype
                if aggr_type.const:
                    member_type = constify(member_type)

                tc.apply_dtype(member_type, expr)

            case PsRel(op1, op2):
                args_tc = TypeContext()
                self.visit_expr(op1, args_tc)
                self.visit_expr(op2, args_tc)

                if args_tc.target_type is None:
                    raise TypificationError(
                        f"Unable to determine type of arguments to relation: {expr}"
                    )
                if not isinstance(args_tc.target_type, PsNumericType):
                    raise TypificationError(
                        f"Invalid type in arguments to relation\n"
                        f"      Expression: {expr}\n"
                        f"  Arguments Type: {args_tc.target_type}"
                    )

                tc.apply_dtype(PsBoolType(), expr)

            case PsBinOp(op1, op2):
                self.visit_expr(op1, tc)
                self.visit_expr(op2, tc)
                tc.infer_dtype(expr)

            case PsNeg(op) | PsNot(op):
                self.visit_expr(op, tc)
                tc.infer_dtype(expr)

            case PsCall(function, args):
                match function:
                    case PsMathFunction():
                        for arg in args:
                            self.visit_expr(arg, tc)
                        tc.infer_dtype(expr)

                    case CFunction(_, arg_types, ret_type):
                        tc.apply_dtype(ret_type, expr)

                        for arg, arg_type in zip(args, arg_types, strict=True):
                            arg_tc = TypeContext(arg_type)
                            self.visit_expr(arg, arg_tc)

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
                        items_tc.apply_dtype(tc.target_type.base_type)
                        tc.infer_dtype(expr)
                else:
                    arr_type = PsArrayType(items_tc.target_type, len(items))
                    tc.apply_dtype(arr_type, expr)

            case PsCast(dtype, arg):
                self.visit_expr(arg, TypeContext())
                tc.apply_dtype(dtype, expr)

            case _:
                raise NotImplementedError(f"Can't typify {expr}")