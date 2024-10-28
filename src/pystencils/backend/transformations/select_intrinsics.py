from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING, cast
from enum import Enum, auto

from ..ast.structural import PsAstNode, PsAssignment, PsStatement
from ..ast.expressions import PsExpression
from ...types import PsVectorType, deconstify
from ..ast.expressions import (
    PsVectorMemAcc,
    PsSymbolExpr,
    PsConstantExpr,
    PsBinOp,
    PsAdd,
    PsSub,
    PsMul,
    PsDiv,
)
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
    def __init__(self) -> None:
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


class MaterializeVectorIntrinsics:
    def __init__(self, platform: GenericVectorCpu):
        self._platform = platform

    def __call__(self, node: PsAstNode) -> PsAstNode:
        return self.visit(node)

    def visit(self, node: PsAstNode) -> PsAstNode:
        match node:
            case PsAssignment(lhs, rhs) if isinstance(lhs, PsVectorMemAcc):
                vc = VecTypeCtx()
                vc.set(lhs.get_vector_type())
                store_arg = self.visit_expr(rhs, vc)
                return PsStatement(self._platform.vector_store(lhs, store_arg))
            case PsExpression():
                return self.visit_expr(node, VecTypeCtx())
            case _:
                node.children = [self(c) for c in node.children]
                return node

    def visit_expr(self, expr: PsExpression, vc: VecTypeCtx) -> PsExpression:
        match expr:
            case PsSymbolExpr(symb):
                if isinstance(symb.dtype, PsVectorType):
                    intrin_type = self._platform.type_intrinsic(symb.dtype)
                    vc.set(symb.dtype)
                    symb.dtype = intrin_type

                return expr

            case PsConstantExpr(c):
                if isinstance(c.dtype, PsVectorType):
                    vc.set(c.dtype)
                    return self._platform.constant_vector(c)
                else:
                    return expr

            case PsVectorMemAcc():
                vc.set(expr.get_vector_type())
                return self._platform.vector_load(expr)

            case PsBinOp(op1, op2):
                op1 = self.visit_expr(op1, vc)
                op2 = self.visit_expr(op2, vc)

                vtype = vc.get()
                if vtype is not None:
                    return self._platform.op_intrinsic(
                        _intrin_op(expr), vtype, [op1, op2]
                    )
                else:
                    return expr

            case expr:
                expr.children = [
                    self.visit_expr(cast(PsExpression, c), vc) for c in expr.children
                ]
                if vc.get() is not None:
                    raise VectorizationError(f"Don't know how to vectorize {expr}")
                return expr


def _intrin_op(expr: PsBinOp) -> IntrinsicOps:
    match expr:
        case PsAdd():
            return IntrinsicOps.ADD
        case PsSub():
            return IntrinsicOps.SUB
        case PsMul():
            return IntrinsicOps.MUL
        case PsDiv():
            return IntrinsicOps.DIV
        case _:
            assert False
