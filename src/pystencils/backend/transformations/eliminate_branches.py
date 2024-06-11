from ..kernelcreation import KernelCreationContext
from ..ast import PsAstNode
from ..ast.analysis import collect_undefined_symbols
from ..ast.structural import PsLoop, PsBlock, PsConditional
from ..ast.expressions import (
    PsAnd,
    PsCast,
    PsConstant,
    PsConstantExpr,
    PsDiv,
    PsEq,
    PsExpression,
    PsGe,
    PsGt,
    PsIntDiv,
    PsLe,
    PsLt,
    PsMul,
    PsNe,
    PsNeg,
    PsNot,
    PsOr,
    PsSub,
    PsSymbolExpr,
    PsAdd,
)

from .eliminate_constants import EliminateConstants
from ...types import PsBoolType, PsIntegerType

__all__ = ["EliminateBranches"]


class IslAnalysisError(Exception):
    """Indicates a fatal error during integer set analysis (based on islpy)"""


class BranchElimContext:
    def __init__(self) -> None:
        self.enclosing_loops: list[PsLoop] = []
        self.enclosing_conditions: list[PsExpression] = []


class EliminateBranches:
    """Replace conditional branches by their then- or else-branch if their condition can be unequivocally
    evaluated.

    This pass will attempt to evaluate branch conditions within their context in the AST, and replace
    conditionals by either their then- or their else-block if the branch is unequivocal.

    If islpy is installed, this pass will incorporate information about the iteration regions
    of enclosing loops and enclosing conditionals into its analysis.

    Args:
        use_isl (bool, optional): enable islpy based analysis (default: True)
    """

    def __init__(self, ctx: KernelCreationContext, use_isl: bool = True) -> None:
        self._ctx = ctx
        self._use_isl = use_isl
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
                    statements_new.append(self.visit(stmt, ec))
                node.statements = statements_new

            case PsConditional():
                result = self.handle_conditional(node, ec)

                match result:
                    case PsConditional(_, branch_true, branch_false):
                        ec.enclosing_conditions.append(result.condition)
                        self.visit(branch_true, ec)
                        ec.enclosing_conditions.pop()

                        if branch_false is not None:
                            ec.enclosing_conditions.append(PsNot(result.condition))
                            self.visit(branch_false, ec)
                            ec.enclosing_conditions.pop()
                    case PsBlock():
                        self.visit(result, ec)
                    case None:
                        result = PsBlock([])
                    case _:
                        assert False, "unreachable code"

                return result

        return node

    def handle_conditional(
        self, conditional: PsConditional, ec: BranchElimContext
    ) -> PsConditional | PsBlock | None:
        condition_simplified = self._elim_constants(conditional.condition)
        if self._use_isl:
            condition_simplified = self._isl_simplify_condition(
                condition_simplified, ec
            )

        match condition_simplified:
            case PsConstantExpr(c) if c.value:
                return conditional.branch_true
            case PsConstantExpr(c) if not c.value:
                return conditional.branch_false

        return conditional

    def _isl_simplify_condition(
        self, condition: PsExpression, ec: BranchElimContext
    ) -> PsExpression:
        """If installed, use ISL to simplify the passed condition to true or
        false based on enclosing loops and conditionals. If no simplification
        can be made or ISL is not installed, the original condition is returned.
        """

        try:
            import islpy as isl
        except ImportError:
            return condition

        def printer(expr: PsExpression):
            match expr:
                case PsSymbolExpr(symbol):
                    return symbol.name

                case PsConstantExpr(constant):
                    dtype = constant.get_dtype()
                    if not isinstance(dtype, (PsIntegerType, PsBoolType)):
                        raise IslAnalysisError(
                            "Only scalar integer and bool constant may appear in isl expressions."
                        )
                    return str(constant.value)

                case PsAdd(op1, op2):
                    return f"({printer(op1)} + {printer(op2)})"
                case PsSub(op1, op2):
                    return f"({printer(op1)} - {printer(op2)})"
                case PsMul(op1, op2):
                    return f"({printer(op1)} * {printer(op2)})"
                case PsDiv(op1, op2) | PsIntDiv(op1, op2):
                    return f"({printer(op1)} / {printer(op2)})"
                case PsAnd(op1, op2):
                    return f"({printer(op1)} and {printer(op2)})"
                case PsOr(op1, op2):
                    return f"({printer(op1)} or {printer(op2)})"
                case PsEq(op1, op2):
                    return f"({printer(op1)} = {printer(op2)})"
                case PsNe(op1, op2):
                    return f"({printer(op1)} != {printer(op2)})"
                case PsGt(op1, op2):
                    return f"({printer(op1)} > {printer(op2)})"
                case PsGe(op1, op2):
                    return f"({printer(op1)} >= {printer(op2)})"
                case PsLt(op1, op2):
                    return f"({printer(op1)} < {printer(op2)})"
                case PsLe(op1, op2):
                    return f"({printer(op1)} <= {printer(op2)})"

                case PsNeg(operand):
                    return f"(-{printer(operand)})"
                case PsNot(operand):
                    return f"(not {printer(operand)})"

                case PsCast(_, operand):
                    return printer(operand)

                case _:
                    raise IslAnalysisError(
                        f"Not supported by isl or don't know how to print {expr}"
                    )

        dofs = collect_undefined_symbols(condition)
        outer_conditions = []

        for loop in ec.enclosing_loops:
            if not (
                isinstance(loop.step, PsConstantExpr)
                and loop.step.constant.value == 1
            ):
                raise IslAnalysisError(
                    "Loops with strides != 1 are not yet supported."
                )

            dofs.add(loop.counter.symbol)
            dofs.update(collect_undefined_symbols(loop.start))
            dofs.update(collect_undefined_symbols(loop.stop))

            loop_start_str = printer(loop.start)
            loop_stop_str = printer(loop.stop)
            ctr_name = loop.counter.symbol.name
            outer_conditions.append(
                f"{ctr_name} >= {loop_start_str} and {ctr_name} < {loop_stop_str}"
            )

        for cond in ec.enclosing_conditions:
            dofs.update(collect_undefined_symbols(cond))
            outer_conditions.append(printer(cond))

        dofs_str = ",".join(dof.name for dof in dofs)
        outer_conditions_str = " and ".join(outer_conditions)
        condition_str = printer(condition)

        outer_set = isl.BasicSet(f"{{ [{dofs_str}] : {outer_conditions_str} }}")
        inner_set = isl.BasicSet(f"{{ [{dofs_str}] : {condition_str} }}")

        if inner_set.is_empty():
            return PsExpression.make(PsConstant(False))

        intersection = outer_set.intersect(inner_set)
        if intersection.is_empty():
            return PsExpression.make(PsConstant(False))
        elif intersection == outer_set:
            return PsExpression.make(PsConstant(True))
        else:
            return condition
