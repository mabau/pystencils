from __future__ import annotations

from pymbolic.mapper.c_code import CCodeMapper

from .ast import (
    ast_visitor,
    PsAstNode,
    PsBlock,
    PsExpression,
    PsDeclaration,
    PsAssignment,
    PsLoop,
    PsConditional,
    PsComment
)
from .ast.kernelfunction import PsKernelFunction
from .typed_expressions import PsTypedVariable
from .functions import Deref, AddressOf, Cast


def emit_code(kernel: PsKernelFunction):
    #   TODO: Specialize for different targets
    printer = CAstPrinter()
    return printer.print(kernel)


class CExpressionsPrinter(CCodeMapper):

    def map_deref(self, deref: Deref, enclosing_prec):
        return "*"
    
    def map_address_of(self, addrof: AddressOf, enclosing_prec):
        return "&"
    
    def map_cast(self, cast: Cast, enclosing_prec):
        return f"({cast.target_type.c_string()})"


class CAstPrinter:
    def __init__(self, indent_width=3):
        self._indent_width = indent_width

        self._current_indent_level = 0

        self._expr_printer = CExpressionsPrinter()

    def indent(self, line):
        return " " * self._current_indent_level + line

    def print(self, node: PsAstNode) -> str:
        return self.visit(node)

    @ast_visitor
    def visit(self, _: PsAstNode) -> str:
        raise ValueError("Cannot print this node.")

    @visit.case(PsKernelFunction)
    def function(self, func: PsKernelFunction) -> str:
        params_spec = func.get_parameters()
        params_str = ", ".join(f"{p.dtype.c_string()} {p.name}" for p in params_spec.params)
        decl = f"FUNC_PREFIX void {func.name} ({params_str})"
        body = self.visit(func.body)
        return f"{decl}\n{body}"

    @visit.case(PsBlock)
    def block(self, block: PsBlock):
        if not block.children:
            return self.indent("{ }")

        self._current_indent_level += self._indent_width
        interior = "\n".join(self.visit(c) for c in block.children)
        self._current_indent_level -= self._indent_width
        return self.indent("{\n") + interior + self.indent("}\n")

    @visit.case(PsExpression)
    def pymb_expression(self, expr: PsExpression):
        return self._expr_printer(expr.expression)

    @visit.case(PsDeclaration)
    def declaration(self, decl: PsDeclaration):
        lhs_symb = decl.declared_variable.symbol
        assert isinstance(lhs_symb, PsTypedVariable)
        lhs_dtype = lhs_symb.dtype
        rhs_code = self.visit(decl.rhs)

        return self.indent(f"{lhs_dtype.c_string()} {lhs_symb.name} = {rhs_code};")

    @visit.case(PsAssignment)
    def assignment(self, asm: PsAssignment):
        lhs_code = self.visit(asm.lhs)
        rhs_code = self.visit(asm.rhs)
        return self.indent(f"{lhs_code} = {rhs_code};\n")

    @visit.case(PsLoop)
    def loop(self, loop: PsLoop):
        ctr_symbol = loop.counter.symbol
        assert isinstance(ctr_symbol, PsTypedVariable)

        ctr = ctr_symbol.name
        start_code = self.visit(loop.start)
        stop_code = self.visit(loop.stop)
        step_code = self.visit(loop.step)

        body_code = self.visit(loop.body)

        code = (
            f"for({ctr_symbol.dtype} {ctr} = {start_code};"
            + f" {ctr} < {stop_code};"
            + f" {ctr} += {step_code})\n"
            + body_code
        )
        return self.indent(code)

    @visit.case(PsConditional)
    def conditional(self, node: PsConditional):
        cond_code = self.visit(node.condition)
        then_code = self.visit(node.branch_true)

        code = f"if({cond_code})\n{then_code}"

        if node.branch_false is not None:
            else_code = self.visit(node.branch_false)
            code += f"\nelse\n{else_code}"

        return self.indent(code)

    @visit.case(PsComment)
    def comment(self, node: PsComment):
        lines = list(node.lines)
        lines[0] = "/* " + lines[0]
        for i in range(1, len(lines)):
            lines[i] = "   " + lines[i]
        lines[-1] = lines[-1] + " */"
        return self.indent("\n".join(lines))
