from typing import overload, cast

import sympy as sp
import pymbolic.primitives as pb
from pymbolic.interop.sympy import SympyToPymbolicMapper
from itertools import chain

from ...assignment import Assignment
from ...simp import AssignmentCollection
from ...field import Field, FieldType
from ...typing import BasicType

from .context import KernelCreationContext

from ..ast.nodes import (
    PsBlock,
    PsAssignment,
    PsDeclaration,
    PsSymbolExpr,
    PsLvalueExpr,
    PsExpression,
)
from ..types import constify, make_type
from ..typed_expressions import PsTypedVariable
from ..arrays import PsArrayAccess
from ..exceptions import PsInputError


class FreezeExpressions(SympyToPymbolicMapper):
    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

    @overload
    def __call__(self, asms: AssignmentCollection) -> PsBlock:
        pass

    @overload
    def __call__(self, expr: sp.Expr) -> PsExpression:
        pass

    @overload
    def __call__(self, asm: Assignment) -> PsAssignment:
        pass

    def __call__(self, obj):
        if isinstance(obj, AssignmentCollection):
            return PsBlock([self.rec(asm) for asm in obj.all_assignments])
        elif isinstance(obj, Assignment):
            return cast(PsAssignment, self.rec(obj))
        elif isinstance(obj, sp.Expr):
            return PsExpression(cast(pb.Expression, self.rec(obj)))
        else:
            raise PsInputError(f"Don't know how to freeze {obj}")

    def map_Assignment(self, expr: Assignment):  # noqa
        lhs = self.rec(expr.lhs)
        rhs = self.rec(expr.rhs)

        if isinstance(lhs, pb.Variable):
            return PsDeclaration(PsSymbolExpr(lhs), PsExpression(rhs))
        elif isinstance(lhs, PsArrayAccess):
            return PsAssignment(PsLvalueExpr(lhs), PsExpression(rhs))
        else:
            assert False, "That should not have happened."

    def map_BasicType(self, expr: BasicType):
        #   TODO: This should not be necessary; the frontend should use the new type system.
        dtype = make_type(expr.numpy_dtype.type)
        if expr.const:
            return constify(dtype)
        else:
            return dtype

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
                        i + o
                        for i, o in zip(
                            self._ctx.get_iteration_space().spatial_indices, offsets
                        )
                    ]
                case FieldType.INDEXED:
                    # flake8: noqa
                    sparse_ispace = self._ctx.get_sparse_iteration_space()
                    #   Add sparse iteration counter to offset
                    assert len(offsets) == 1 # must have been checked by the context
                    offsets = [offsets[0] + sparse_ispace.sparse_counter]
                case FieldType.CUSTOM:
                    raise ValueError("Custom fields support only absolute accesses.")
                case unknown:
                    raise NotImplementedError(
                        f"Cannot translate accesses to field type {unknown} yet."
                    )

        summands = tuple(
            idx * stride
            for idx, stride in zip(offsets + indices, array.strides, strict=True)
        )

        index = summands[0] if len(summands) == 1 else pb.Sum(summands)

        return PsArrayAccess(ptr, index)
    
    def map_Function(self, func: sp.Function):
        """Map a SymPy function to a backend-supported function symbol.

        SymPy functions are frozen to an instance of `nbackend.functions.PsFunction`.
        """
        raise NotImplementedError()
