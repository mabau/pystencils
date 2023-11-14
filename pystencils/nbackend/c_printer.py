from __future__ import annotations

from pymbolic.mapper.c_code import CCodeMapper

from .ast import ast_visitor, PsAstNode, PsBlock, PsExpression, PsDeclaration, PsAssignment, PsLoop


class CPrinter:
    def __init__(self, indent_width=3):
        self._indent_width = indent_width

        self._current_indent_level = 0
        self._inside_expression = False  # controls parentheses in nested arithmetic expressions

        self._pb_cmapper = CCodeMapper()

    def indent(self, line):
        return " " * self._current_indent_level + line

    def print(self, node: PsAstNode):
        return self.visit(node)

    @ast_visitor
    def visit(self, node: PsAstNode):
        raise ValueError("Cannot print this node.")

    @visit.case(PsBlock)
    def block(self, block: PsBlock):
        if not block.children:
            return self.indent("{ }")

        self._current_indent_level += self._indent_width
        interior = "".join(self.visit(c) for c in block.children)
        self._current_indent_level -= self._indent_width
        return self.indent("{\n") + interior + self.indent("}\n")

    @visit.case(PsExpression)
    def pymb_expression(self, expr: PsExpression):
        return self._pb_cmapper(expr.expression)

    @visit.case(PsDeclaration)
    def declaration(self, decl: PsDeclaration):
        lhs_symb = decl.lhs.symbol
        lhs_dtype = lhs_symb.dtype
        rhs_code = self.visit(decl.rhs)

        return self.indent(f"{lhs_dtype} {lhs_symb.name} = {rhs_code};\n")

    @visit.case(PsAssignment)
    def assignment(self, asm: PsAssignment):
        lhs_code = self.visit(asm.lhs)
        rhs_code = self.visit(asm.rhs)
        return self.indent(f"{lhs_code} = {rhs_code};\n")

    @visit.case(PsLoop)
    def loop(self, loop: PsLoop):
        ctr_symbol = loop.counter.expression
        ctr = ctr_symbol.name
        start_code = self.visit(loop.start)
        stop_code = self.visit(loop.stop)
        step_code = self.visit(loop.step)

        body_code = self.visit(loop.body)

        code = f"for({ctr_symbol.dtype} {ctr} = {start_code};" + \
               f" {ctr} < {stop_code};" + \
               f" {ctr} += {step_code})\n" + \
               body_code
        return code
