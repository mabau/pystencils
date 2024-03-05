from typing import overload, cast
from functools import reduce
from operator import add, mul

import sympy as sp

from ...sympyextensions import Assignment, AssignmentCollection
from ...sympyextensions.typed_sympy import TypedSymbol
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
)

from ..constants import PsConstant
from ...types import PsStructType
from ..exceptions import PsInputError
from ..functions import PsMathFunction, MathFunctions


class FreezeError(Exception):
    """Signifies an error during expression freezing."""


class FreezeExpressions:
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

    def visit_expr(self, expr: sp.Basic):
        if not isinstance(expr, sp.Expr):
            raise FreezeError(f"Cannot freeze {expr} to an expression")
        return cast(PsExpression, self.visit(expr))

    def freeze_expression(self, expr: sp.Expr) -> PsExpression:
        return cast(PsExpression, self.visit(expr))

    def map_Assignment(self, expr: Assignment):  # noqa
        lhs = self.visit(expr.lhs)
        rhs = self.visit(expr.rhs)

        assert isinstance(lhs, PsExpression)
        assert isinstance(rhs, PsExpression)

        if isinstance(lhs, PsSymbolExpr):
            return PsDeclaration(lhs, rhs)
        elif isinstance(lhs, (PsArrayAccess, PsVectorArrayAccess)):  # todo
            return PsAssignment(lhs, rhs)
        else:
            assert False, "That should not have happened."

    def map_Symbol(self, spsym: sp.Symbol) -> PsSymbolExpr:
        symb = self._ctx.get_symbol(spsym.name)
        return PsSymbolExpr(symb)

    def map_Add(self, expr: sp.Add) -> PsExpression:
        return reduce(add, (self.visit_expr(arg) for arg in expr.args))

    def map_Mul(self, expr: sp.Mul) -> PsExpression:
        return reduce(mul, (self.visit_expr(arg) for arg in expr.args))

    def map_Integer(self, expr: sp.Integer) -> PsConstantExpr:
        value = int(expr)
        return PsConstantExpr(PsConstant(value))

    def map_Rational(self, expr: sp.Rational) -> PsExpression:
        num = PsConstantExpr(PsConstant(expr.numerator))
        denom = PsConstantExpr(PsConstant(expr.denominator))
        return num / denom

    def map_TypedSymbol(self, expr: TypedSymbol):
        dtype = expr.dtype
        symb = self._ctx.get_symbol(expr.name, dtype)
        return PsSymbolExpr(symb)

    def map_Access(self, access: Field.Access):
        field = access.field
        array = self._ctx.get_array(field)
        ptr = array.base_pointer

        offsets: list[PsExpression] = [self.visit_expr(o) for o in access.offsets]
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
            indices = [self.visit_expr(i) for i in access.index]
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
            case sp.Min():
                func_symbol = PsMathFunction(MathFunctions.Min)
            case sp.Max():
                func_symbol = PsMathFunction(MathFunctions.Max)
            case _:
                raise FreezeError(f"Unsupported function: {func}")

        args = tuple(self.visit_expr(arg) for arg in func.args)
        return PsCall(func_symbol, args)
