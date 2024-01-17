import pymbolic.primitives as pb
from pymbolic.interop.sympy import SympyToPymbolicMapper

from ...field import Field, FieldType

from .context import KernelCreationContext, IterationSpace, SparseIterationSpace

from ..ast.nodes import PsAssignment
from ..types import PsSignedIntegerType, PsIeeeFloatType, PsUnsignedIntegerType
from ..typed_expressions import PsTypedVariable
from ..arrays import PsArrayAccess
from ..exceptions import PsInternalCompilerError


class FreezeExpressions(SympyToPymbolicMapper):
    def __init__(self, ctx: KernelCreationContext, ispace: IterationSpace):
        self._ctx = ctx
        self._ispace = ispace

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
        array_desc = self._ctx.get_array_descriptor(field)
        array = array_desc.array
        ptr = array_desc.base_ptr

        offsets: list[pb.Expression] = [self.rec(o) for o in access.offsets]
        indices: list[pb.Expression] = [self.rec(o) for o in access.index]

        if not access.is_absolute_access:
            match field.field_type:
                case FieldType.GENERIC:
                    #   Add the iteration counters
                    offsets = [
                        i + o for i, o in zip(self._ispace.spatial_indices, offsets)
                    ]
                case FieldType.INDEXED:
                    if isinstance(self._ispace, SparseIterationSpace):
                        #   TODO: make sure index (and all offsets?) are zero
                        #   TODO: Add sparse iteration counter
                        raise NotImplementedError()
                    else:
                        raise PsInternalCompilerError(
                            "Cannot translate index field access without a sparse iteration space."
                        )
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
