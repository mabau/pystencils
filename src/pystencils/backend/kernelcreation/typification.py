from __future__ import annotations

from typing import TypeVar, Callable

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
    PsScalarType,
    PsVectorType,
    constify,
    deconstify,
)
from ..ast.structural import (
    PsAstNode,
    PsBlock,
    PsLoop,
    PsConditional,
    PsExpression,
    PsAssignment,
    PsDeclaration,
    PsStatement,
    PsEmptyLeafMixIn,
)
from ..ast.expressions import (
    PsBufferAcc,
    PsArrayInitList,
    PsBinOp,
    PsIntOpTrait,
    PsNumericOpTrait,
    PsBoolOpTrait,
    PsCall,
    PsTernary,
    PsCast,
    PsAddressOf,
    PsConstantExpr,
    PsLookup,
    PsSubscript,
    PsMemAcc,
    PsSymbolExpr,
    PsLiteralExpr,
    PsRel,
    PsNeg,
    PsNot,
)
from ..ast.vector import PsVecBroadcast, PsVecMemAcc
from ..functions import PsMathFunction, CFunction
from ..ast.util import determine_memory_object

__all__ = ["Typifier"]


class TypificationError(Exception):
    """Indicates a fatal error during typification."""


NodeT = TypeVar("NodeT", bound=PsAstNode)

ResolutionHook = Callable[[PsType], None]


class TypeContext:
    """Typing context, with support for type inference and checking.

    Instances of this class are used to propagate and check data types across expression subtrees
    of the AST. Each type context has a target type `target_type`, which shall be applied to all expressions it covers
    """

    def __init__(
        self,
        target_type: PsType | None = None,
    ):
        self._deferred_exprs: list[PsExpression] = []

        self._target_type = deconstify(target_type) if target_type is not None else None

        self._hooks: list[ResolutionHook] = []

    @property
    def target_type(self) -> PsType | None:
        return self._target_type

    def add_hook(self, hook: ResolutionHook):
        """Adds a resolution hook to this context.

        The hook will be called with the context's target type as soon as it becomes known,
        which might be immediately.
        """
        if self._target_type is None:
            self._hooks.append(hook)
        else:
            hook(self._target_type)

    def apply_dtype(self, dtype: PsType, expr: PsExpression | None = None):
        """Applies the given ``dtype`` to this type context, and optionally to the given expression.

        If the context's target_type is already known, it must be compatible with the given dtype.
        If the target type is still unknown, target_type is set to dtype and retroactively applied
        to all deferred expressions.

        If an expression is specified, it will be covered by the type context.
        If the expression already has a data type set, it must be compatible with the target type
        and will be replaced by it.
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
        assert self._target_type is not None

        for hook in self._hooks:
            hook(self._target_type)
        self._hooks = []

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
                    if symb.dtype is None:
                        #   Symbols are not forced to constness
                        symb.dtype = deconstify(self._target_type)
                    elif not self._compatible(symb.dtype):
                        raise TypificationError(
                            f"Type mismatch at symbol {symb}: Symbol type did not match the context's target type\n"
                            f"    Symbol type: {symb.dtype}\n"
                            f"    Target type: {self._target_type}"
                        )

                case PsNumericOpTrait() if not isinstance(
                    self._target_type, PsNumericType
                ) or self._target_type.is_bool():
                    #   FIXME: PsBoolType derives from PsNumericType, but is not numeric
                    raise TypificationError(
                        f"Numerical operation encountered in non-numerical type context:\n"
                        f"    Expression: {expr}"
                        f"  Type Context: {self._target_type}"
                    )

                case PsIntOpTrait() if not (
                    isinstance(self._target_type, PsNumericType)
                    and self._target_type.is_int()
                ):
                    raise TypificationError(
                        f"Integer operation encountered in non-integer type context:\n"
                        f"    Expression: {expr}"
                        f"  Type Context: {self._target_type}"
                    )

                case PsBoolOpTrait() if not (
                    isinstance(self._target_type, PsNumericType)
                    and self._target_type.is_bool()
                ):
                    raise TypificationError(
                        f"Boolean operation encountered in non-boolean type context:\n"
                        f"    Expression: {expr}"
                        f"  Type Context: {self._target_type}"
                    )
        # endif
        expr.dtype = self._target_type

    def _compatible(self, dtype: PsType):
        """Checks whether the given data type is compatible with the context's target type.
        The two must match except for constness.
        """
        assert self._target_type is not None
        return deconstify(dtype) == self._target_type


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

     - The context's ``default_dtype`` is applied to all untyped symbols encountered inside a right-hand side expression
     - If an untyped symbol is encountered on an assignment's left-hand side, it will first be attempted to infer its
       type from the right-hand side. If that fails, the context's ``default_dtype`` will be applied.
     - It is an error if an untyped symbol occurs in the same type context as a typed symbol or constant
       with a non-default data type.
     - By default, all expressions receive a ``const`` type unless they occur on a (non-declaration) assignment's
       left-hand side

    **Typing of symbol expressions**

    Some expressions (`PsSymbolExpr`, `PsArrayAccess`) encapsulate symbols and inherit their data types.
    """

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

    def __call__(self, node: NodeT) -> NodeT:
        if isinstance(node, PsExpression):
            tc = TypeContext()
            self.visit_expr(node, tc)

            if tc.target_type is None:
                #   no type could be inferred -> take the default
                tc.apply_dtype(self._ctx.default_dtype)
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

            case PsStatement(expr):
                tc = TypeContext()
                self.visit_expr(expr, tc)
                if tc.target_type is None:
                    tc.apply_dtype(self._ctx.default_dtype)

            case PsDeclaration(lhs, rhs) if isinstance(rhs, PsArrayInitList):
                #   Special treatment for array declarations
                assert isinstance(lhs, PsSymbolExpr)

                decl_tc = TypeContext()
                items_tc = TypeContext()

                if (lhs_type := lhs.symbol.dtype) is not None:
                    if not isinstance(lhs_type, PsArrayType):
                        raise TypificationError(
                            f"Illegal LHS type in array declaration: {lhs_type}"
                        )

                    if lhs_type.shape != rhs.shape:
                        raise TypificationError(
                            f"Incompatible shapes in declaration of array symbol {lhs.symbol}.\n"
                            f"  Symbol shape: {lhs_type.shape}\n"
                            f"   Array shape: {rhs.shape}"
                        )

                    items_tc.apply_dtype(lhs_type.base_type)
                    decl_tc.apply_dtype(lhs_type, lhs)
                else:
                    decl_tc.infer_dtype(lhs)

                for item in rhs.items:
                    self.visit_expr(item, items_tc)

                if items_tc.target_type is None:
                    items_tc.apply_dtype(self._ctx.default_dtype)

                if decl_tc.target_type is None:
                    assert items_tc.target_type is not None
                    decl_tc.apply_dtype(
                        PsArrayType(items_tc.target_type, rhs.shape), rhs
                    )
                else:
                    decl_tc.infer_dtype(rhs)

            case PsDeclaration(lhs, rhs) | PsAssignment(lhs, rhs):
                #   Only if the LHS is an untyped symbol, infer its type from the RHS
                infer_lhs = isinstance(lhs, PsSymbolExpr) and lhs.symbol.dtype is None

                tc = TypeContext()

                if infer_lhs:
                    tc.infer_dtype(lhs)
                else:
                    self.visit_expr(lhs, tc)
                    assert tc.target_type is not None

                self.visit_expr(rhs, tc)

                if infer_lhs and tc.target_type is None:
                    #   no type has been inferred -> use the default dtype
                    tc.apply_dtype(self._ctx.default_dtype)
                elif not isinstance(node, PsDeclaration):
                    #   check mutability of LHS
                    _, lhs_const = determine_memory_object(lhs)
                    if lhs_const:
                        raise TypificationError(f"Cannot assign to immutable LHS {lhs}")

            case PsConditional(cond, branch_true, branch_false):
                cond_tc = TypeContext(PsBoolType())
                self.visit_expr(cond, cond_tc)

                self.visit(branch_true)

                if branch_false is not None:
                    self.visit(branch_false)

            case PsLoop(ctr, start, stop, step, body):
                if ctr.symbol.dtype is None:
                    ctr.symbol.apply_dtype(self._ctx.index_dtype)
                    ctr.dtype = ctr.symbol.get_dtype()

                tc_index = TypeContext(ctr.symbol.dtype)
                self.visit_expr(start, tc_index)
                self.visit_expr(stop, tc_index)
                self.visit_expr(step, tc_index)

                self.visit(body)

            case PsEmptyLeafMixIn():
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
            case PsSymbolExpr(symb):
                if symb.dtype is None:
                    symb.dtype = self._ctx.default_dtype
                tc.apply_dtype(symb.dtype, expr)

            case PsConstantExpr(c):
                if c.dtype is not None:
                    tc.apply_dtype(c.dtype, expr)
                else:
                    tc.infer_dtype(expr)

            case PsLiteralExpr(lit):
                tc.apply_dtype(lit.dtype, expr)

            case PsBufferAcc(_, indices):
                tc.apply_dtype(expr.buffer.element_type, expr)
                for idx in indices:
                    self._handle_idx(idx)

            case PsMemAcc(ptr, offset) | PsVecMemAcc(ptr, offset):
                ptr_tc = TypeContext()
                self.visit_expr(ptr, ptr_tc)

                if not isinstance(ptr_tc.target_type, PsPointerType):
                    raise TypificationError(
                        f"Type of pointer argument to memory access was not a pointer type: {ptr_tc.target_type}"
                    )

                tc.apply_dtype(ptr_tc.target_type.base_type, expr)
                self._handle_idx(offset)

                if isinstance(expr, PsVecMemAcc) and expr.stride is not None:
                    self._handle_idx(expr.stride)

            case PsSubscript(arr, indices):
                if isinstance(arr, PsArrayInitList):
                    shape = arr.shape

                    #   extend outer context over the init-list entries
                    for item in arr.items:
                        self.visit_expr(item, tc)

                    #   learn the array type from the items
                    def arr_hook(element_type: PsType):
                        arr.dtype = PsArrayType(element_type, arr.shape)

                    tc.add_hook(arr_hook)
                else:
                    #   type of array has to be known
                    arr_tc = TypeContext()
                    self.visit_expr(arr, arr_tc)

                    if not isinstance(arr_tc.target_type, PsArrayType):
                        raise TypificationError(
                            f"Type of array argument to subscript was not an array type: {arr_tc.target_type}"
                        )

                    tc.apply_dtype(arr_tc.target_type.base_type, expr)
                    shape = arr_tc.target_type.shape

                if len(indices) != len(shape):
                    raise TypificationError(
                        f"Invalid number of indices to {len(shape)}-dimensional array: {len(indices)}"
                    )

                for idx in indices:
                    self._handle_idx(idx)

            case PsAddressOf(arg):
                if not isinstance(
                    arg, (PsSymbolExpr, PsSubscript, PsMemAcc, PsBufferAcc, PsLookup)
                ):
                    raise TypificationError(
                        f"Illegal expression below AddressOf operator: {arg}"
                    )

                arg_tc = TypeContext()
                self.visit_expr(arg, arg_tc)

                if arg_tc.target_type is None:
                    raise TypificationError(
                        f"Unable to determine type of argument to AddressOf: {arg}"
                    )

                #   Inherit pointed-to type from referenced object, not from the subexpression
                match arg:
                    case PsSymbolExpr(s):
                        pointed_to_type = s.get_dtype()
                    case PsSubscript(ptr, _) | PsMemAcc(ptr, _) | PsBufferAcc(ptr, _):
                        arr_type = ptr.get_dtype()
                        assert isinstance(arr_type, PsDereferencableType)
                        pointed_to_type = arr_type.base_type
                    case PsLookup(aggr, member_name):
                        struct_type = aggr.get_dtype()
                        assert isinstance(struct_type, PsStructType)
                        if struct_type.const:
                            pointed_to_type = constify(
                                struct_type.get_member(member_name).dtype
                            )
                        else:
                            pointed_to_type = deconstify(
                                struct_type.get_member(member_name).dtype
                            )
                    case _:
                        assert False, "unreachable code"

                ptr_type = PsPointerType(pointed_to_type, const=True)
                tc.apply_dtype(ptr_type, expr)

            case PsLookup(aggr, member_name):
                #   Members of a struct type inherit the struct type's `const` qualifier
                aggr_tc = TypeContext()
                self.visit_expr(aggr, aggr_tc)
                aggr_type = aggr_tc.target_type

                if not isinstance(aggr_type, PsStructType):
                    raise TypificationError(
                        "Aggregate type of lookup is not a struct type."
                    )

                member = aggr_type.find_member(member_name)
                if member is None:
                    raise TypificationError(
                        f"Aggregate of type {aggr_type} does not have a member {member_name}."
                    )

                member_type = member.dtype
                if aggr_type.const:
                    member_type = constify(member_type)

                tc.apply_dtype(member_type, expr)

            case PsTernary(cond, then, els):
                cond_tc = TypeContext(target_type=PsBoolType())
                self.visit_expr(cond, cond_tc)

                self.visit_expr(then, tc)
                self.visit_expr(els, tc)
                tc.infer_dtype(expr)

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

                if isinstance(args_tc.target_type, PsVectorType):
                    tc.apply_dtype(
                        PsVectorType(PsBoolType(), args_tc.target_type.vector_entries),
                        expr,
                    )
                else:
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

            case PsArrayInitList(_):
                raise TypificationError(
                    "Unable to typify array initializer in isolation.\n"
                    f"    Array: {expr}"
                )

            case PsCast(dtype, arg):
                arg_tc = TypeContext()
                self.visit_expr(arg, arg_tc)

                if arg_tc.target_type is None:
                    raise TypificationError(
                        f"Unable to determine type of argument to Cast: {arg}"
                    )

                tc.apply_dtype(dtype, expr)

            case PsVecBroadcast(lanes, arg):
                op_tc = TypeContext()
                self.visit_expr(arg, op_tc)

                if op_tc.target_type is None:
                    raise TypificationError(
                        f"Unable to determine type of argument to vector broadcast: {arg}"
                    )

                if not isinstance(op_tc.target_type, PsScalarType):
                    raise TypificationError(
                        f"Illegal type in argument to vector broadcast: {op_tc.target_type}"
                    )

                tc.apply_dtype(PsVectorType(op_tc.target_type, lanes), expr)

            case _:
                raise NotImplementedError(f"Can't typify {expr}")

    def _handle_idx(self, idx: PsExpression):
        index_tc = TypeContext()
        self.visit_expr(idx, index_tc)

        if index_tc.target_type is None:
            index_tc.apply_dtype(self._ctx.index_dtype, idx)
        elif not isinstance(index_tc.target_type, PsIntegerType):
            raise TypificationError(
                f"Invalid data type in index expression.\n"
                f"    Expression: {idx}\n"
                f"          Type: {index_tc.target_type}"
            )
