from ..kernelcreation import KernelCreationContext
from ..ast import PsAstNode
from ..ast.structural import PsLoop, PsBlock, PsConditional
from ..ast.expressions import PsConstantExpr

from .eliminate_constants import EliminateConstants

__all__ = ["EliminateBranches"]


class BranchElimContext:
    def __init__(self) -> None:
        self.enclosing_loops: list[PsLoop] = []


class EliminateBranches:
    """Replace conditional branches by their then- or else-branch if their condition can be unequivocally
    evaluated.

    This pass will attempt to evaluate branch conditions within their context in the AST, and replace
    conditionals by either their then- or their else-block if the branch is unequivocal.

    TODO: If islpy is installed, this pass will incorporate information about the iteration regions
    of enclosing loops into its analysis.
    """

    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx
        self._elim_constants = EliminateConstants(ctx, extract_constant_exprs=False)

    def __call__(self, node: PsAstNode) -> PsAstNode:
        return self.visit(node, BranchElimContext())

    def visit(self, node: PsAstNode, ec: BranchElimContext) -> PsAstNode:
        match node:
            case PsLoop(_, _, _, _, body):
                ec.enclosing_loops.append(node)
                self.visit(body, ec)
                ec.enclosing_loops.pop()

            case PsBlock(statements):
                statements_new: list[PsAstNode] = []
                for stmt in statements:
                    if isinstance(stmt, PsConditional):
                        result = self.handle_conditional(stmt, ec)
                        if result is not None:
                            statements_new.append(result)
                    else:
                        statements_new.append(self.visit(stmt, ec))
                node.statements = statements_new

            case PsConditional():
                result = self.handle_conditional(node, ec)
                if result is None:
                    return PsBlock([])
                else:
                    return result

        return node

    def handle_conditional(
        self, conditional: PsConditional, ec: BranchElimContext
    ) -> PsConditional | PsBlock | None:
        condition_simplified = self._elim_constants(conditional.condition)
        match condition_simplified:
            case PsConstantExpr(c) if c.value:
                return conditional.branch_true
            case PsConstantExpr(c) if not c.value:
                return conditional.branch_false

        #   TODO: Analyze condition against counters of enclosing loops using ISL

        return conditional
