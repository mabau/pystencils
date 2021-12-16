import logging
from typing import List, Union

import sympy as sp
from sympy.codegen import Assignment
from sympy.codegen.rewriting import ReplaceOptim, optimize

from pystencils.astnodes import Node

from pystencils.functions import DivFunc


class NodeCollection:
    def __init__(self, assignments: List[Union[Node, Assignment]]):
        self.all_assignments = assignments

        if all((isinstance(a, Assignment) for a in assignments)):
            self.is_Nodes = False
            self.is_Assignments = True
        elif all((isinstance(n, Node) for n in assignments)):
            self.is_Nodes = True
            self.is_Assignments = False
            logging.warning('Using Nodes is experimental and not fully tested. Double check your generated code!')
        else:
            raise ValueError(f'The list "{assignments}" is mixed. Pass either a list of "pystencils.Assignments" '
                             f'or a list of "pystencils.astnodes.Node')

        self.simplification_hints = ()

    def evaluate_terms(self):

        # There is no visitor implemented now so working with nodes does not work
        if self.is_Nodes:
            return

        evaluate_constant_terms = ReplaceOptim(
            lambda e: hasattr(e, 'is_constant') and e.is_constant and not e.is_integer,
            lambda p: p.evalf())

        evaluate_pow = ReplaceOptim(
            lambda e: e.is_Pow and e.exp.is_Integer and abs(e.exp) <= 8,
            lambda p: (
                sp.UnevaluatedExpr(sp.Mul(*([p.base] * +p.exp), evaluate=False)) if p.exp > 0 else
                DivFunc(sp.Integer(1), sp.Mul(*([p.base] * -p.exp), evaluate=False))
            ))

        sympy_optimisations = [evaluate_constant_terms, evaluate_pow]
        self.all_assignments = [Assignment(a.lhs, optimize(a.rhs, sympy_optimisations))
                                if hasattr(a, 'lhs')
                                else a for a in self.all_assignments]
