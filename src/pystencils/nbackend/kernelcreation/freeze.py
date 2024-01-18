import pymbolic.primitives as pb
from pymbolic.interop.sympy import SympyToPymbolicMapper

from ...field import Field, FieldType

from .context import KernelCreationContext

from ..ast.nodes import PsAssignment
from ..types import PsSignedIntegerType, PsIeeeFloatType, PsUnsignedIntegerType
from ..typed_expressions import PsTypedVariable
from ..arrays import PsArrayAccess


class FreezeExpressions(SympyToPymbolicMapper):
    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

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
            raise NotImplementedError("Data type not supported.")

    def map_FieldShapeSymbol(self, expr):
        dtype = self.rec(expr.dtype)
        return PsTypedVariable(expr.name, dtype)

    def map_TypedSymbol(self, expr):
        dtype = self.rec(expr.dtype)
        return PsTypedVariable(expr.name, dtype)

    def map_Access(self, access: Field.Access):
        field = access.field
        array = self._ctx.get_array(field)
        ptr = array.base_pointer

        offsets: list[pb.Expression] = [self.rec(o) for o in access.offsets]
        indices: list[pb.Expression] = [self.rec(o) for o in access.index]

        if not access.is_absolute_access:
            match field.field_type:
                case FieldType.GENERIC:
                    #   Add the iteration counters
                    offsets = [
                        i + o for i, o in zip(self._ctx.get_iteration_space().spatial_indices, offsets)
                    ]
                case FieldType.INDEXED:
                    # flake8: noqa
                    sparse_ispace = self._ctx.get_sparse_iteration_space()
                    #   TODO: make sure index (and all offsets?) are zero
                    #   TODO: Add sparse iteration counter
                    raise NotImplementedError()
                case FieldType.CUSTOM:
                    raise ValueError("Custom fields support only absolute accesses.")
                case unknown:
                    raise NotImplementedError(
                        f"Cannot translate accesses to field type {unknown} yet."
                    )

        index = pb.Sum(
            tuple(
                idx * stride
                for idx, stride in zip(offsets + indices, array.strides, strict=True)
            )
        )

        return PsArrayAccess(ptr, index)
