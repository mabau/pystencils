from collections.abc import Iterable
from typing import Any, Dict, List, Union, Optional, Set

import sympy
import sympy as sp
from sympy.codegen.ast import Assignment, AddAugmentedAssignment
from sympy.codegen.rewriting import ReplaceOptim, optimize

from pystencils.astnodes import Block, Node, SympyAssignment
from pystencils.backends.cbackend import CustomCodeNode
from pystencils.functions import DivFunc
from pystencils.simp import AssignmentCollection


class NodeCollection:
    def __init__(self, assignments: List[Union[Node, Assignment]],
                 simplification_hints: Optional[Dict[str, Any]] = None,
                 bound_fields: Set[sp.Symbol] = None, rhs_fields: Set[sp.Symbol] = None):
        nodes = list()
        assignments = [assignments, ] if not isinstance(assignments, Iterable) else assignments
        for assignment in assignments:
            if isinstance(assignment, Assignment):
                nodes.append(SympyAssignment(assignment.lhs, assignment.rhs))
            elif isinstance(assignment, AddAugmentedAssignment):
                nodes.append(SympyAssignment(assignment.lhs, assignment.lhs + assignment.rhs))
            elif isinstance(assignment, Node):
                nodes.append(assignment)
            else:
                raise ValueError(f"Unknown node in the AssignmentCollection: {assignment}")

        self.all_assignments = nodes
        self.simplification_hints = simplification_hints if simplification_hints else {}
        self.bound_fields = bound_fields if bound_fields else {}
        self.rhs_fields = rhs_fields if rhs_fields else {}

    @staticmethod
    def from_assignment_collection(assignment_collection: AssignmentCollection):
        return NodeCollection(assignments=assignment_collection.all_assignments,
                              simplification_hints=assignment_collection.simplification_hints,
                              bound_fields=assignment_collection.bound_fields,
                              rhs_fields=assignment_collection.rhs_fields)

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

        def visitor(node):
            if isinstance(node, CustomCodeNode):
                return node
            elif isinstance(node, Block):
                return node.func([visitor(child) for child in node.args])
            elif isinstance(node, SympyAssignment):
                new_lhs = visitor(node.lhs)
                new_rhs = visitor(node.rhs)
                return node.func(new_lhs, new_rhs, node.is_const, node.use_auto)
            elif isinstance(node, Node):
                return node.func(*[visitor(child) for child in node.args])
            elif isinstance(node, sympy.Basic):
                return optimize(node, sympy_optimisations)
            else:
                raise NotImplementedError(f'{node} {type(node)} has no valid visitor')

        self.all_assignments = [visitor(assignment) for assignment in self.all_assignments]
