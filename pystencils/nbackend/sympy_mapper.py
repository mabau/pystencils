from pymbolic.interop.sympy import SympyToPymbolicMapper

from pystencils.typing import TypedSymbol
from pystencils.typing.typed_sympy import SHAPE_DTYPE
from .ast.nodes import PsAssignment, PsSymbolExpr
from .types import PsSignedIntegerType, PsIeeeFloatType, PsUnsignedIntegerType
from .typed_expressions import PsTypedVariable
from .arrays import PsArrayBasePointer, PsLinearizedArray, PsArrayAccess

CTR_SYMBOLS = [TypedSymbol(f"ctr_{i}", SHAPE_DTYPE) for i in range(3)]


class PystencilsToPymbolicMapper(SympyToPymbolicMapper):
    def map_Assignment(self, expr):  # noqa
        lhs = self.rec(expr.lhs)
        rhs = self.rec(expr.rhs)
        return PsAssignment(lhs, rhs)

    def map_BasicType(self, expr):
        width = expr.numpy_dtype.itemsize * 8
        const = expr.const
        if expr.is_float():
            return PsIeeeFloatType(width, const)
        elif expr.is_uint():
            return PsUnsignedIntegerType(width, const)
        elif expr.is_int():
            return PsSignedIntegerType(width, const)
        else:
            raise (NotImplementedError, "Not supported dtype")

    def map_FieldShapeSymbol(self, expr):
        dtype = self.rec(expr.dtype)
        return PsTypedVariable(expr.name, dtype)

    def map_TypedSymbol(self, expr):
        dtype = self.rec(expr.dtype)
        return PsTypedVariable(expr.name, dtype)

    def map_Access(self, expr):
        name = expr.field.name
        shape = tuple([self.rec(s) for s in expr.field.shape])
        strides = tuple([self.rec(s) for s in expr.field.strides])
        dtype = self.rec(expr.dtype)

        array = PsLinearizedArray(name, shape, strides, dtype)

        ptr = PsArrayBasePointer(expr.name, array)
        index = sum(
            [ctr * stride for ctr, stride in zip(CTR_SYMBOLS, expr.field.strides)]
        )
        index = self.rec(index)

        return PsSymbolExpr(PsArrayAccess(ptr, index))
