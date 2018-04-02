# -*- coding: utf-8 -*-
from sympy.codegen.ast import Assignment
from sympy.printing.latex import LatexPrinter

__all__ = ['Assignment']


def print_assignment_latex(printer, expr):
    """sympy cannot print Assignments as Latex. Thus, this function is added to the sympy Latex printer"""
    printed_lhs = printer.doprint(expr.lhs)
    printed_rhs = printer.doprint(expr.rhs)
    return f"{printed_lhs} \leftarrow {printed_rhs}"


def assignment_str(assignment):
    return f"{assignment.lhs} ← {assignment.rhs}"


Assignment.__str__ = assignment_str
LatexPrinter._print_Assignment = print_assignment_latex
