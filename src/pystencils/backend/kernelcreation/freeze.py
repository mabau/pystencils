from typing import overload, cast, Any
from functools import reduce
from operator import add, mul, sub

import sympy as sp

from ...sympyextensions import Assignment, AssignmentCollection
from ...sympyextensions.typed_sympy import TypedSymbol, CastFunc
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
    PsArrayAccess,
    PsVectorArrayAccess,
    PsLookup,
    PsCall,
    PsConstantExpr,
    PsArrayInitList,
    PsSubscript,
    PsCast,
)

from ..constants import PsConstant
from ...types import PsStructType
from ..exceptions import PsInputError
from ..functions import PsMathFunction, MathFunctions


class FreezeError(Exception):
    """Signifies an error during expression freezing."""


class FreezeExpressions:
    """Convert expressions and kernels expressed in the SymPy language to the code generator's internal representation.

    This class accepts a subset of the SymPy symbolic algebra language complemented with the extensions
    implemented in `pystencils.sympyextensions`, and converts it to the abstract syntax tree representation
    of the pystencils code generator. It is invoked early during the code generation process.

    TODO: Document the full set of supported SymPy features, with restrictions and caveats
    TODO: Properly document the SymPy extensions provided by pystencils

    TODO: This is a (possibly incomplete) list of SymPy language features that still need to be implemented:

     - Augmented Assignments
     - AddressOf
     - Relations (sp.Relational)
     - pystencils.sympyextensions.integer_functions
     - pystencils.sympyextensions.bit_masks
     - GPU fast approximations (pystencils.fast_approximation)
     - ConditionalFieldAccess
     - sp.Piecewise
     - sp.floor, sp.ceiling
     - sp.log, sp.atan2, sp.sinh, sp.cosh. sp.atan
     - sp.Min, sp.Max: multi-argument versions
     - Modulus (sp.Mod)

    """

    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

    @overload
    def __call__(self, obj: AssignmentCollection) -> PsBlock:
        pass

    @overload
    def __call__(self, obj: sp.Expr) -> PsExpression:
        pass

    @overload
    def __call__(self, obj: Assignment) -> PsAssignment:
        pass

    def __call__(self, obj: AssignmentCollection | sp.Basic) -> PsAstNode:
        if isinstance(obj, AssignmentCollection):
            return PsBlock([self.visit(asm) for asm in obj.all_assignments])
        elif isinstance(obj, Assignment):
            return cast(PsAssignment, self.visit(obj))
        elif isinstance(obj, sp.Expr):
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

    def visit_expr_like(self, obj: Any) -> PsExpression:
        if isinstance(obj, sp.Basic):
            return self.visit_expr(obj)
        elif isinstance(obj, (int, float, bool)):
            return PsExpression.make(PsConstant(obj))
        else:
            raise FreezeError(f"Don't know how to freeze {obj}")

    def visit_expr(self, expr: sp.Basic):
        if not isinstance(expr, (sp.Expr, sp.Tuple)):
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
        elif isinstance(lhs, (PsArrayAccess, PsLookup, PsVectorArrayAccess)):  # todo
            return PsAssignment(lhs, rhs)
        else:
            raise FreezeError(
                f"Encountered unsupported expression on assignment left-hand side: {lhs}"
            )

    def map_Symbol(self, spsym: sp.Symbol) -> PsSymbolExpr:
        symb = self._ctx.get_symbol(spsym.name)
        return PsSymbolExpr(symb)

    def map_Add(self, expr: sp.Add) -> PsExpression:
        #   TODO: think about numerically sensible ways of freezing sums and products
        signs: list[int] = []
        for summand in expr.args:
            if summand.is_negative:
                signs.append(-1)
            elif isinstance(summand, sp.Mul) and any(
                factor.is_negative for factor in summand.args
            ):
                signs.append(-1)
            else:
                signs.append(1)

        frozen_expr = self.visit_expr(expr.args[0])

        for sign, arg in zip(signs[1:], expr.args[1:]):
            if sign == -1:
                arg = -arg
                op = sub
            else:
                op = add

            frozen_expr = op(frozen_expr, self.visit_expr(arg))

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
        symb = self._ctx.get_symbol(expr.name, dtype)
        return PsSymbolExpr(symb)

    def map_Tuple(self, expr: sp.Tuple) -> PsArrayInitList:
        items = [self.visit_expr(item) for item in expr]
        return PsArrayInitList(items)

    def map_Indexed(self, expr: sp.Indexed) -> PsSubscript:
        assert isinstance(expr.base, sp.IndexedBase)
        base = self.visit_expr(expr.base.label)
        subscript = PsSubscript(base, self.visit_expr(expr.indices[0]))
        for idx in expr.indices[1:]:
            subscript = PsSubscript(subscript, self.visit_expr(idx))
        return subscript

    def map_Access(self, access: Field.Access):
        field = access.field
        array = self._ctx.get_array(field)
        ptr = array.base_pointer

        offsets: list[PsExpression] = [self.visit_expr_like(o) for o in access.offsets]
        indices: list[PsExpression]

        if not access.is_absolute_access:
            match field.field_type:
                case FieldType.GENERIC:
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
                case FieldType.CUSTOM:
                    raise ValueError("Custom fields support only absolute accesses.")
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
            indices = [self.visit_expr_like(i) for i in access.index]
            if not indices:
                # For canonical representation, there must always be at least one index dimension
                indices = [PsExpression.make(PsConstant(0))]

        summands = tuple(
            idx * PsExpression.make(stride)
            for idx, stride in zip(offsets + indices, array.strides, strict=True)
        )

        index = summands[0] if len(summands) == 1 else reduce(add, summands)

        if struct_member_name is not None:
            # Produce a Lookup here, don't check yet if the member name is valid. That's the typifier's job.
            return PsLookup(PsArrayAccess(ptr, index), struct_member_name)
        else:
            return PsArrayAccess(ptr, index)

    def map_Function(self, func: sp.Function) -> PsCall:
        """Map SymPy function calls by mapping sympy function classes to backend-supported function symbols.

        SymPy functions are frozen to an instance of `nbackend.functions.PsFunction`.
        """
        match func:
            case sp.Abs():
                func_symbol = PsMathFunction(MathFunctions.Abs)
            case sp.exp():
                func_symbol = PsMathFunction(MathFunctions.Exp)
            case sp.sin():
                func_symbol = PsMathFunction(MathFunctions.Sin)
            case sp.cos():
                func_symbol = PsMathFunction(MathFunctions.Cos)
            case sp.tan():
                func_symbol = PsMathFunction(MathFunctions.Tan)
            case _:
                raise FreezeError(f"Unsupported function: {func}")

        args = tuple(self.visit_expr(arg) for arg in func.args)
        return PsCall(func_symbol, args)

    def map_Min(self, expr: sp.Min) -> PsCall:
        args = tuple(self.visit_expr(arg) for arg in expr.args)
        return PsCall(PsMathFunction(MathFunctions.Min), args)

    def map_Max(self, expr: sp.Max) -> PsCall:
        args = tuple(self.visit_expr(arg) for arg in expr.args)
        return PsCall(PsMathFunction(MathFunctions.Max), args)

    def map_CastFunc(self, cast_expr: CastFunc):
        return PsCast(cast_expr.dtype, self.visit_expr(cast_expr.expr))
