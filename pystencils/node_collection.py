from typing import List, Union

import sympy
import sympy as sp
from sympy.codegen import Assignment
from sympy.codegen.rewriting import ReplaceOptim, optimize

from pystencils.astnodes import Block, Node, SympyAssignment
from pystencils.backends.cbackend import CustomCodeNode
from pystencils.functions import DivFunc
from pystencils.simp import AssignmentCollection


class NodeCollection:
    def __init__(self, assignments: List[Union[Node, Assignment]]):
        self.all_assignments = assignments

        if all((isinstance(a, Assignment) for a in assignments)):
            self.is_Nodes = False
            self.is_Assignments = True
        elif all((isinstance(n, Node) for n in assignments)):
            self.is_Nodes = True
            self.is_Assignments = False
        else:
            raise ValueError(f'The list "{assignments}" is mixed. Pass either a list of "pystencils.Assignments" '
                             f'or a list of "pystencils.astnodes.Node')

        self.simplification_hints = {}

    @staticmethod
    def from_assignment_collection(assignment_collection: AssignmentCollection):
        nodes = list()
        for assignemt in assignment_collection.all_assignments:
            if isinstance(assignemt, Assignment):
                nodes.append(SympyAssignment(assignemt.lhs, assignemt.rhs))
            elif isinstance(assignemt, Node):
                nodes.append(assignemt)
            else:
                raise ValueError(f"Unknown node in the AssignmentCollection: {assignemt}")

        return NodeCollection(nodes)

    def evaluate_terms(self):
        evaluate_constant_terms = ReplaceOptim(
            lambda e: hasattr(e, 'is_constant') and e.is_constant and not e.is_integer,
            lambda p: p.evalf()
        )

        evaluate_pow = ReplaceOptim(
            lambda e: e.is_Pow and e.exp.is_Integer and abs(e.exp) <= 8,
            lambda p: sp.UnevaluatedExpr(sp.Mul(*([p.base] * +p.exp), evaluate=False)) if p.exp > 0 else
            (DivFunc(sp.Integer(1), p.base) if p.exp == -1 else
             DivFunc(sp.Integer(1), sp.UnevaluatedExpr(sp.Mul(*([p.base] * -p.exp), evaluate=False))))
        )
        sympy_optimisations = [evaluate_constant_terms, evaluate_pow]

        if self.is_Nodes:
            def visitor(node):
                if isinstance(node, CustomCodeNode):
                    return node
                elif isinstance(node, Block):
                    return node.func([visitor(child) for child in node.args])
                elif isinstance(node, Node):
                    return node.func(*[visitor(child) for child in node.args])
                elif isinstance(node, sympy.Basic):
                    return optimize(node, sympy_optimisations)
                else:
                    raise NotImplementedError(f'{node} {type(node)} has no valid visitor')

            self.all_assignments = [visitor(assignment) for assignment in self.all_assignments]
        else:
            self.all_assignments = [Assignment(a.lhs, optimize(a.rhs, sympy_optimisations))
                                    if hasattr(a, 'lhs')
                                    else a for a in self.all_assignments]
