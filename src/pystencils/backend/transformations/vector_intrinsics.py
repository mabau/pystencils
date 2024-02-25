from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING
from enum import Enum, auto

import pymbolic.primitives as pb
from pymbolic.mapper import IdentityMapper

from ..ast import PsAstNode, PsExpression, PsAssignment, PsStatement
from ..types import PsVectorType, deconstify
from ..typed_expressions import PsTypedVariable, PsTypedConstant, ExprOrConstant
from ..arrays import PsVectorArrayAccess
from ..exceptions import PsInternalCompilerError

if TYPE_CHECKING:
    from ..platforms import GenericVectorCpu


__all__ = ["IntrinsicOps", "MaterializeVectorIntrinsics"]

NodeT = TypeVar("NodeT", bound=PsAstNode)


class IntrinsicOps(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    FMA = auto()


class VectorizationError(Exception):
    """Exception indicating a fatal error during vectorization."""


class VecTypeCtx:
    def __init__(self):
        self._dtype: None | PsVectorType = None

    def get(self) -> PsVectorType | None:
        return self._dtype

    def set(self, dtype: PsVectorType):
        dtype = deconstify(dtype)
        if self._dtype is not None and dtype != self._dtype:
            raise PsInternalCompilerError(
                f"Ambiguous vector types: {self._dtype} and {dtype}"
            )
        self._dtype = dtype

    def reset(self):
        self._dtype = None


class MaterializeVectorIntrinsics(IdentityMapper):
    def __init__(self, platform: GenericVectorCpu):
        self._platform = platform

    def __call__(self, node: PsAstNode) -> PsAstNode:
        match node:
            case PsExpression(expr):
                # descend into expr
                node.expression = self.rec(expr, VecTypeCtx())
                return node
            case PsAssignment(lhs, rhs) if isinstance(
                lhs.expression, PsVectorArrayAccess
            ):
                vc = VecTypeCtx()
                vc.set(lhs.expression.dtype)
                store_arg = self.rec(rhs.expression, vc)
                return PsStatement(
                    PsExpression(self._platform.vector_store(lhs.expression, store_arg))
                )
            case other:
                other.children = (self(c) for c in other.children)
        return node

    def map_typed_variable(
        self, tv: PsTypedVariable, vc: VecTypeCtx
    ) -> PsTypedVariable:
        if isinstance(tv.dtype, PsVectorType):
            intrin_type = self._platform.type_intrinsic(tv.dtype)
            vc.set(tv.dtype)
            return PsTypedVariable(tv.name, intrin_type)
        else:
            return tv

    def map_constant(self, c: PsTypedConstant, vc: VecTypeCtx) -> ExprOrConstant:
        if isinstance(c.dtype, PsVectorType):
            vc.set(c.dtype)
            return self._platform.constant_vector(c)
        else:
            return c

    def map_vector_array_access(
        self, acc: PsVectorArrayAccess, vc: VecTypeCtx
    ) -> pb.Expression:
        vc.set(acc.dtype)
        return self._platform.vector_load(acc)

    def map_sum(self, expr: pb.Sum, vc: VecTypeCtx) -> pb.Expression:
        args = [self.rec(arg, vc) for arg in expr.children]
        vtype = vc.get()
        if vtype is not None:
            if len(args) != 2:
                raise VectorizationError("Cannot vectorize non-binary sums")
            return self._platform.op_intrinsic(IntrinsicOps.ADD, vtype, args)
        else:
            return expr

    def map_product(self, expr: pb.Product, vc: VecTypeCtx) -> pb.Expression:
        args = [self.rec(arg, vc) for arg in expr.children]
        vtype = vc.get()
        if vtype is not None:
            if len(args) != 2:
                raise VectorizationError("Cannot vectorize non-binary products")
            return self._platform.op_intrinsic(IntrinsicOps.MUL, vtype, args)
        else:
            return expr
