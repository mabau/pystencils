from typing import overload, cast, Any
from functools import reduce
from operator import add, mul, sub, truediv

import sympy as sp
import sympy.core.relational
import sympy.logic.boolalg
from sympy.codegen.ast import AssignmentBase, AugmentedAssignment

from ...assignment import Assignment
from ...simp import AssignmentCollection
from ...sympyextensions import (
    integer_functions,
    ConditionalFieldAccess,
)
from ...sympyextensions.typed_sympy import TypedSymbol, CastFunc, DynamicType
from ...sympyextensions.pointers import AddressOf, mem_acc
from ...field import Field, FieldType

from .context import KernelCreationContext

from ..ast.structural import (
    PsAstNode,
    PsBlock,
    PsAssignment,
    PsDeclaration,
    PsExpression,
    PsSymbolExpr,
)
from ..ast.expressions import (
    PsBufferAcc,
    PsArrayInitList,
    PsBitwiseAnd,
    PsBitwiseOr,
    PsBitwiseXor,
    PsAddressOf,
    PsCall,
    PsCast,
    PsConstantExpr,
    PsIntDiv,
    PsRem,
    PsLeftShift,
    PsLookup,
    PsRightShift,
    PsSubscript,
    PsTernary,
    PsRel,
    PsEq,
    PsNe,
    PsLt,
    PsGt,
    PsLe,
    PsGe,
    PsAnd,
    PsOr,
    PsNot,
    PsMemAcc
)
from ..ast.vector import PsVecMemAcc

from ..constants import PsConstant
from ...types import PsNumericType, PsStructType, PsType
from ..exceptions import PsInputError
from ..functions import PsMathFunction, MathFunctions


class FreezeError(Exception):
    """Signifies an error during expression freezing."""


ExprLike = (
    sp.Expr
    | sp.Tuple
    | sympy.core.relational.Relational
    | sympy.logic.boolalg.BooleanFunction
)
_ExprLike = (
    sp.Expr,
    sp.Tuple,
    sympy.core.relational.Relational,
    sympy.logic.boolalg.BooleanFunction,
)


class FreezeExpressions:
    """Convert expressions and kernels expressed in the SymPy language to the code generator's internal representation.

    This class accepts a subset of the SymPy symbolic algebra language complemented with the extensions
    implemented in `pystencils.sympyextensions`, and converts it to the abstract syntax tree representation
    of the pystencils code generator. It is invoked early during the code generation process.

    TODO: Document the full set of supported SymPy features, with restrictions and caveats
    TODO: Properly document the SymPy extensions provided by pystencils
    """

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

    @overload
    def __call__(self, obj: AssignmentCollection) -> PsBlock:
        pass

    @overload
    def __call__(self, obj: ExprLike) -> PsExpression:
        pass

    @overload
    def __call__(self, obj: AssignmentBase) -> PsAssignment:
        pass

    def __call__(self, obj: AssignmentCollection | sp.Basic) -> PsAstNode:
        if isinstance(obj, AssignmentCollection):
            return PsBlock([self.visit(asm) for asm in obj.all_assignments])
        elif isinstance(obj, AssignmentBase):
            return cast(PsAssignment, self.visit(obj))
        elif isinstance(obj, _ExprLike):
            return cast(PsExpression, self.visit(obj))
        else:
            raise PsInputError(f"Don't know how to freeze {obj}")

    def visit(self, node: sp.Basic) -> PsAstNode:
        mro = list(type(node).__mro__)

        while mro:
            method_name = "map_" + mro.pop(0).__name__

            try:
                method = self.__getattribute__(method_name)
            except AttributeError:
                pass
            else:
                return method(node)

        raise FreezeError(f"Don't know how to freeze expression {node}")

    def visit_expr_or_builtin(self, obj: Any) -> PsExpression:
        if isinstance(obj, _ExprLike):
            return self.visit_expr(obj)
        elif isinstance(obj, (int, float, bool)):
            return PsExpression.make(PsConstant(obj))
        else:
            raise FreezeError(f"Don't know how to freeze {obj}")

    def visit_expr(self, expr: sp.Basic):
        if not isinstance(expr, _ExprLike):
            raise FreezeError(f"Cannot freeze {expr} to an expression")
        return cast(PsExpression, self.visit(expr))

    def freeze_expression(self, expr: sp.Expr) -> PsExpression:
        return cast(PsExpression, self.visit(expr))

    def map_Assignment(self, expr: Assignment):
        lhs = self.visit(expr.lhs)
        rhs = self.visit(expr.rhs)

        assert isinstance(lhs, PsExpression)
        assert isinstance(rhs, PsExpression)

        if isinstance(lhs, PsSymbolExpr):
            return PsDeclaration(lhs, rhs)
        elif isinstance(lhs, (PsBufferAcc, PsLookup, PsVecMemAcc)):
            return PsAssignment(lhs, rhs)
        else:
            raise FreezeError(
                f"Encountered unsupported expression on assignment left-hand side: {lhs}"
            )

    def map_AugmentedAssignment(self, expr: AugmentedAssignment):
        lhs = self.visit(expr.lhs)
        rhs = self.visit(expr.rhs)

        assert isinstance(lhs, PsExpression)
        assert isinstance(rhs, PsExpression)

        match expr.op:
            case "+=":
                op = add
            case "-=":
                op = sub
            case "*=":
                op = mul
            case "/=":
                op = truediv
            case _:
                raise FreezeError(f"Unsupported augmented assignment: {expr.op}.")

        return PsAssignment(lhs, op(lhs.clone(), rhs))

    def map_Symbol(self, spsym: sp.Symbol) -> PsSymbolExpr:
        symb = self._ctx.get_symbol(spsym.name)
        return PsSymbolExpr(symb)

    def map_Add(self, expr: sp.Add) -> PsExpression:
        #   TODO: think about numerically sensible ways of freezing sums and products

        frozen_expr = self.visit_expr(expr.args[0])

        for summand in expr.args[1:]:
            if isinstance(summand, sp.Mul) and any(
                factor == -1 for factor in summand.args
            ):
                summand = -summand
                op = sub
            else:
                op = add

            frozen_expr = op(frozen_expr, self.visit_expr(summand))

        return frozen_expr

    def map_Mul(self, expr: sp.Mul) -> PsExpression:
        return reduce(mul, (self.visit_expr(arg) for arg in expr.args))

    def map_Pow(self, expr: sp.Pow) -> PsExpression:
        base = expr.args[0]
        exponent = expr.args[1]

        base_frozen = self.visit_expr(base)
        reciprocal = False
        expand_product = False

        if exponent.is_Integer:
            if exponent == 0:
                return PsExpression.make(PsConstant(1))

            if exponent.is_negative:
                reciprocal = True
                exponent = -exponent

            if exponent <= sp.Integer(
                5
            ):  # TODO: is this a sensible limit? maybe make this configurable.
                expand_product = True

        if expand_product:
            frozen_expr = reduce(
                mul,
                [base_frozen]
                + [base_frozen.clone() for _ in range(0, int(exponent) - 1)],
            )
        else:
            exponent_frozen = self.visit_expr(exponent)
            frozen_expr = PsMathFunction(MathFunctions.Pow)(
                base_frozen, exponent_frozen
            )

        if reciprocal:
            one = PsExpression.make(PsConstant(1))
            frozen_expr = one / frozen_expr

        return frozen_expr

    def map_Integer(self, expr: sp.Integer) -> PsConstantExpr:
        value = int(expr)
        return PsConstantExpr(PsConstant(value))

    def map_Float(self, expr: sp.Float) -> PsConstantExpr:
        value = float(expr)  # TODO: check accuracy of evaluation
        return PsConstantExpr(PsConstant(value))

    def map_Rational(self, expr: sp.Rational) -> PsExpression:
        num = PsConstantExpr(PsConstant(expr.numerator))
        denom = PsConstantExpr(PsConstant(expr.denominator))
        return num / denom

    def map_TypedSymbol(self, expr: TypedSymbol):
        dtype = expr.dtype

        match dtype:
            case DynamicType.NUMERIC_TYPE:
                dtype = self._ctx.default_dtype
            case DynamicType.INDEX_TYPE:
                dtype = self._ctx.index_dtype

        symb = self._ctx.get_symbol(expr.name, dtype)
        return PsSymbolExpr(symb)

    def map_Tuple(self, expr: sp.Tuple) -> PsArrayInitList:
        if not expr:
            raise FreezeError("Cannot translate an empty tuple.")

        items = [self.visit_expr(item) for item in expr]
        
        if any(isinstance(i, PsArrayInitList) for i in items):
            #  base case: have nested arrays
            if not all(isinstance(i, PsArrayInitList) for i in items):
                raise FreezeError(
                    f"Cannot translate nested arrays of non-uniform shape: {expr}"
                )
            
            subarrays = cast(list[PsArrayInitList], items)
            shape_tail = subarrays[0].shape
            
            if not all(s.shape == shape_tail for s in subarrays[1:]):
                raise FreezeError(
                    f"Cannot translate nested arrays of non-uniform shape: {expr}"
                )
            
            return PsArrayInitList([s.items_grid for s in subarrays])  # type: ignore
        else:
            #  base case: no nested arrays
            return PsArrayInitList(items)

    def map_Indexed(self, expr: sp.Indexed) -> PsSubscript:
        assert isinstance(expr.base, sp.IndexedBase)
        base = self.visit_expr(expr.base.label)
        indices = [self.visit_expr(i) for i in expr.indices]
        return PsSubscript(base, indices)

    def map_Access(self, access: Field.Access):
        field = access.field
        array = self._ctx.get_buffer(field)
        ptr = array.base_pointer

        offsets: list[PsExpression] = [
            self.visit_expr_or_builtin(o) for o in access.offsets
        ]
        indices: list[PsExpression]

        if not access.is_absolute_access:
            match field.field_type:
                case FieldType.GENERIC | FieldType.CUSTOM:
                    #   Add the iteration counters
                    offsets = [
                        PsExpression.make(i) + o
                        for i, o in zip(
                            self._ctx.get_iteration_space().spatial_indices, offsets
                        )
                    ]
                case FieldType.INDEXED:
                    sparse_ispace = self._ctx.get_sparse_iteration_space()
                    #   Add sparse iteration counter to offset
                    assert len(offsets) == 1  # must have been checked by the context
                    offsets = [
                        offsets[0] + PsExpression.make(sparse_ispace.sparse_counter)
                    ]
                case FieldType.BUFFER:
                    ispace = self._ctx.get_full_iteration_space()
                    compressed_ctr = ispace.compressed_counter()
                    assert len(offsets) == 1
                    offsets = [compressed_ctr + offsets[0]]
                case unknown:
                    raise NotImplementedError(
                        f"Cannot translate accesses to field type {unknown} yet."
                    )

        #   If the array type is a struct, accesses are modelled using strings
        if isinstance(array.element_type, PsStructType):
            if isinstance(access.index, str):
                struct_member_name = access.index
                indices = [PsExpression.make(PsConstant(0))]
            elif len(access.index) == 1 and isinstance(access.index[0], str):
                struct_member_name = access.index[0]
                indices = [PsExpression.make(PsConstant(0))]
            else:
                raise FreezeError(
                    f"Unsupported access into field with struct-type elements: {access}"
                )
        else:
            struct_member_name = None
            indices = [self.visit_expr_or_builtin(i) for i in access.index]
            if not indices:
                # For canonical representation, there must always be at least one index dimension
                indices = [PsExpression.make(PsConstant(0))]

        if struct_member_name is not None:
            # Produce a Lookup here, don't check yet if the member name is valid. That's the typifier's job.
            return PsLookup(PsBufferAcc(ptr, offsets + indices), struct_member_name)
        else:
            return PsBufferAcc(ptr, offsets + indices)

    def map_ConditionalFieldAccess(self, acc: ConditionalFieldAccess):
        facc = self.visit_expr(acc.access)
        condition = self.visit_expr(acc.outofbounds_condition)
        fallback = self.visit_expr(acc.outofbounds_value)
        return PsTernary(condition, fallback, facc)

    def map_Function(self, func: sp.Function) -> PsExpression:
        """Map SymPy function calls by mapping sympy function classes to backend-supported function symbols.

        If applicable, functions are mapped to binary operators, e.g. `backend.ast.expressions.PsBitwiseXor`.
        Other SymPy functions are frozen to an instance of `nbackend.functions.PsFunction`.
        """
        args = tuple(self.visit_expr(arg) for arg in func.args)

        match func:
            case sp.Abs():
                return PsCall(PsMathFunction(MathFunctions.Abs), args)
            case sp.floor():
                return PsCall(PsMathFunction(MathFunctions.Floor), args)
            case sp.ceiling():
                return PsCall(PsMathFunction(MathFunctions.Ceil), args)
            case sp.exp():
                return PsCall(PsMathFunction(MathFunctions.Exp), args)
            case sp.log():
                return PsCall(PsMathFunction(MathFunctions.Log), args)
            case sp.sin():
                return PsCall(PsMathFunction(MathFunctions.Sin), args)
            case sp.cos():
                return PsCall(PsMathFunction(MathFunctions.Cos), args)
            case sp.tan():
                return PsCall(PsMathFunction(MathFunctions.Tan), args)
            case sp.sinh():
                return PsCall(PsMathFunction(MathFunctions.Sinh), args)
            case sp.cosh():
                return PsCall(PsMathFunction(MathFunctions.Cosh), args)
            case sp.asin():
                return PsCall(PsMathFunction(MathFunctions.ASin), args)
            case sp.acos():
                return PsCall(PsMathFunction(MathFunctions.ACos), args)
            case sp.atan():
                return PsCall(PsMathFunction(MathFunctions.ATan), args)
            case sp.atan2():
                return PsCall(PsMathFunction(MathFunctions.ATan2), args)
            case integer_functions.int_div():
                return PsIntDiv(*args)
            case integer_functions.int_rem():
                return PsRem(*args)
            case integer_functions.bit_shift_left():
                return PsLeftShift(*args)
            case integer_functions.bit_shift_right():
                return PsRightShift(*args)
            case integer_functions.bitwise_and():
                return PsBitwiseAnd(*args)
            case integer_functions.bitwise_xor():
                return PsBitwiseXor(*args)
            case integer_functions.bitwise_or():
                return PsBitwiseOr(*args)
            case integer_functions.int_power_of_2():
                return PsLeftShift(PsExpression.make(PsConstant(1)), args[0])
            case integer_functions.round_to_multiple_towards_zero():
                return PsIntDiv(args[0], args[1]) * args[1]
            case integer_functions.ceil_to_multiple():
                return (
                    PsIntDiv(
                        args[0] + args[1] - PsExpression.make(PsConstant(1)), args[1]
                    )
                    * args[1]
                )
            case integer_functions.div_ceil():
                return PsIntDiv(
                    args[0] + args[1] - PsExpression.make(PsConstant(1)), args[1]
                )
            case AddressOf():
                return PsAddressOf(*args)
            case mem_acc():
                return PsMemAcc(*args)
            case _:
                raise FreezeError(f"Unsupported function: {func}")

    def map_Piecewise(self, expr: sp.Piecewise) -> PsTernary:
        from sympy.functions.elementary.piecewise import ExprCondPair

        cases: list[ExprCondPair] = cast(list[ExprCondPair], expr.args)

        if cases[-1].cond != sp.true:
            raise FreezeError(
                "The last case of a `Piecewise` must be the fallback case, its condition must always be `True`."
            )

        conditions = [self.visit_expr(c.cond) for c in cases[:-1]]
        subexprs = [self.visit_expr(c.expr) for c in cases]

        last_expr = subexprs.pop()
        ternary = PsTernary(conditions.pop(), subexprs.pop(), last_expr)

        while conditions:
            ternary = PsTernary(conditions.pop(), subexprs.pop(), ternary)

        return ternary

    def map_Min(self, expr: sp.Min) -> PsCall:
        return self._minmax(expr, PsMathFunction(MathFunctions.Min))

    def map_Max(self, expr: sp.Max) -> PsCall:
        return self._minmax(expr, PsMathFunction(MathFunctions.Max))

    def _minmax(self, expr: sp.Min | sp.Max, func: PsMathFunction) -> PsCall:
        args = [self.visit_expr(arg) for arg in expr.args]
        while len(args) > 1:
            args = [
                (PsCall(func, (args[i], args[i + 1])) if i + 1 < len(args) else args[i])
                for i in range(0, len(args), 2)
            ]
        return cast(PsCall, args[0])

    def map_CastFunc(self, cast_expr: CastFunc) -> PsCast | PsConstantExpr:
        dtype: PsType
        match cast_expr.dtype:
            case DynamicType.NUMERIC_TYPE:
                dtype = self._ctx.default_dtype
            case DynamicType.INDEX_TYPE:
                dtype = self._ctx.index_dtype
            case other if isinstance(other, PsType):
                dtype = other

        arg = self.visit_expr(cast_expr.expr)
        if (
            isinstance(arg, PsConstantExpr)
            and arg.constant.dtype is None
            and isinstance(dtype, PsNumericType)
        ):
            # As of now, the typifier can not infer the type of a bare constant.
            # However, untyped constants may not appear in ASTs from which
            # kernel functions are generated. Therefore, we annotate constants
            # instead of casting them.
            return PsConstantExpr(arg.constant.interpret_as(dtype))
        else:
            return PsCast(dtype, arg)

    def map_Relational(self, rel: sympy.core.relational.Relational) -> PsRel:
        arg1, arg2 = [self.visit_expr(arg) for arg in rel.args]
        match rel.rel_op:  # type: ignore
            case "==":
                return PsEq(arg1, arg2)
            case "!=":
                return PsNe(arg1, arg2)
            case ">=":
                return PsGe(arg1, arg2)
            case "<=":
                return PsLe(arg1, arg2)
            case ">":
                return PsGt(arg1, arg2)
            case "<":
                return PsLt(arg1, arg2)
            case other:
                raise FreezeError(f"Unsupported relation: {other}")

    def map_And(self, conj: sympy.logic.And) -> PsAnd:
        args = [self.visit_expr(arg) for arg in conj.args]
        return reduce(PsAnd, args)  # type: ignore

    def map_Or(self, disj: sympy.logic.Or) -> PsOr:
        args = [self.visit_expr(arg) for arg in disj.args]
        return reduce(PsOr, args)  # type: ignore

    def map_Not(self, neg: sympy.logic.Not) -> PsNot:
        arg = self.visit_expr(neg.args[0])
        return PsNot(arg)
