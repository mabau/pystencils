from typing import Any, Dict, List, Union, Optional, Set

import sympy
import sympy as sp
from sympy.codegen.rewriting import ReplaceOptim, optimize

from pystencils.assignment import Assignment, AddAugmentedAssignment
import pystencils.astnodes as ast
from pystencils.backends.cbackend import CustomCodeNode
from pystencils.functions import DivFunc
from pystencils.simp import AssignmentCollection
from pystencils.typing import FieldPointerSymbol


class NodeCollection:
    def __init__(self, assignments: List[Union[ast.Node, Assignment]],
                 simplification_hints: Optional[Dict[str, Any]] = None,
                 bound_fields: Set[sp.Symbol] = None, rhs_fields: Set[sp.Symbol] = None):

        def visit(obj):
            if isinstance(obj, (list, tuple)):
                return [visit(e) for e in obj]
            if isinstance(obj, Assignment):
                if isinstance(obj.lhs, FieldPointerSymbol):
                    return ast.SympyAssignment(obj.lhs, obj.rhs, is_const=obj.lhs.dtype.const)
                return ast.SympyAssignment(obj.lhs, obj.rhs)
            elif isinstance(obj, AddAugmentedAssignment):
                return ast.SympyAssignment(obj.lhs, obj.lhs + obj.rhs)
            elif isinstance(obj, ast.SympyAssignment):
                return obj
            elif isinstance(obj, ast.Conditional):
                true_block = visit(obj.true_block)
                false_block = None if obj.false_block is None else visit(obj.false_block)
                return ast.Conditional(obj.condition_expr, true_block=true_block, false_block=false_block)
            elif isinstance(obj, ast.Block):
                return ast.Block([visit(e) for e in obj.args])
            elif isinstance(obj, ast.Node) and not isinstance(obj, ast.LoopOverCoordinate):
                return obj
            else:
                raise ValueError("Invalid object in the List of Assignments " + str(type(obj)))

        self.all_assignments = visit(assignments)
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
            elif isinstance(node, ast.Block):
                return node.func([visitor(child) for child in node.args])
            elif isinstance(node, ast.SympyAssignment):
                new_lhs = visitor(node.lhs)
                new_rhs = visitor(node.rhs)
                return node.func(new_lhs, new_rhs, node.is_const, node.use_auto)
            elif isinstance(node, ast.Node):
                return node.func(*[visitor(child) for child in node.args])
            elif isinstance(node, sympy.Basic):
                return optimize(node, sympy_optimisations)
            else:
                raise NotImplementedError(f'{node} {type(node)} has no valid visitor')

        self.all_assignments = [visitor(assignment) for assignment in self.all_assignments]
