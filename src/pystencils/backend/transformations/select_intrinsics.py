from __future__ import annotations
from typing import cast

from ..kernelcreation import KernelCreationContext
from ..memory import PsSymbol
from ..ast.structural import PsAstNode, PsDeclaration, PsAssignment, PsStatement
from ..ast.expressions import PsExpression
from ...types import PsVectorType, constify, deconstify
from ..ast.expressions import PsSymbolExpr, PsConstantExpr, PsUnOp, PsBinOp
from ..ast.vector import PsVecMemAcc
from ..exceptions import MaterializationError

from ..platforms import GenericVectorCpu


__all__ = ["SelectIntrinsics"]


class SelectionContext:
    def __init__(self, ctx: KernelCreationContext, platform: GenericVectorCpu):
        self._ctx = ctx
        self._platform = platform
        self._intrin_symbols: dict[PsSymbol, PsSymbol] = dict()
        self._lane_mask: PsSymbol | None = None

    def get_intrin_symbol(self, symb: PsSymbol) -> PsSymbol:
        if symb not in self._intrin_symbols:
            assert isinstance(symb.dtype, PsVectorType)
            intrin_type = self._platform.type_intrinsic(deconstify(symb.dtype))

            if symb.dtype.const:
                intrin_type = constify(intrin_type)

            replacement = self._ctx.duplicate_symbol(symb, intrin_type)
            self._intrin_symbols[symb] = replacement

        return self._intrin_symbols[symb]


class SelectIntrinsics:
    """Lower IR vector types to intrinsic vector types, and IR vector operations to intrinsic vector operations.
    
    This transformation will replace all vectorial IR elements by conforming implementations using
    compiler intrinsics for the given execution platform.

    Args:
        ctx: The current kernel creation context
        platform: Platform object representing the target hardware, which provides the intrinsics

    Raises:
        MaterializationError: If a vector type or operation cannot be represented by intrinsics
            on the given platform
    """

    def __init__(self, ctx: KernelCreationContext, platform: GenericVectorCpu):
        self._ctx = ctx
        self._platform = platform

    def __call__(self, node: PsAstNode) -> PsAstNode:
        return self.visit(node, SelectionContext(self._ctx, self._platform))

    def visit(self, node: PsAstNode, sc: SelectionContext) -> PsAstNode:
        match node:
            case PsExpression() if isinstance(node.dtype, PsVectorType):
                return self.visit_expr(node, sc)

            case PsDeclaration(lhs, rhs) if isinstance(lhs.dtype, PsVectorType):
                lhs_new = cast(PsSymbolExpr, self.visit_expr(lhs, sc))
                rhs_new = self.visit_expr(rhs, sc)
                return PsDeclaration(lhs_new, rhs_new)
            
            case PsAssignment(lhs, rhs) if isinstance(lhs, PsVecMemAcc): 
                new_rhs = self.visit_expr(rhs, sc)
                return PsStatement(self._platform.vector_store(lhs, new_rhs))
            
            case _:
                node.children = [self.visit(c, sc) for c in node.children]

        return node

    def visit_expr(self, expr: PsExpression, sc: SelectionContext) -> PsExpression:
        if not isinstance(expr.dtype, PsVectorType):
            return expr

        match expr:
            case PsSymbolExpr(symb):
                return PsSymbolExpr(sc.get_intrin_symbol(symb))

            case PsConstantExpr(c):
                return self._platform.constant_intrinsic(c)

            case PsUnOp(operand):
                op = self.visit_expr(operand, sc)
                return self._platform.op_intrinsic(expr, [op])

            case PsBinOp(operand1, operand2):
                op1 = self.visit_expr(operand1, sc)
                op2 = self.visit_expr(operand2, sc)

                return self._platform.op_intrinsic(expr, [op1, op2])

            case PsVecMemAcc():
                return self._platform.vector_load(expr)

            case _:
                raise MaterializationError(
                    f"Unable to select intrinsic implementation for {expr}"
                )
