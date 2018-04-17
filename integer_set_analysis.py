"""Transformations using integer sets based on ISL library"""

import sympy as sp
import islpy as isl
from typing import Tuple

import pystencils.astnodes as ast
from pystencils.transformations import parents_of_type

#context = isl.Context()

"""
- find all Condition nodes
- check if they depend on integers only
- create ISL space containing names of all loop symbols (counter and bounds) and all integers in Conditional expression
- build up pre-condition set by iteration over each enclosing loop add ISL constraints
- build up ISL space for condition
- if pre_condition_set.intersect(conditional_set) == pre_condition_set
    always use True condition
  elif pre_condition_set.intersect(conditional_set).is_empty():
    always use False condition
"""


def isl_iteration_set(node: ast.Node):
    """Builds up an ISL set describing the iteration space by analysing the enclosing loops of the given node. """
    conditions = []
    loop_symbols = set()
    for loop in parents_of_type(node, ast.LoopOverCoordinate):
        if loop.step != 1:
            raise NotImplementedError("Loops with strides != 1 are not yet supported.")

        loop_symbols.add(loop.loop_counter_symbol)
        loop_symbols.update(sp.sympify(loop.start).atoms(sp.Symbol))
        loop_symbols.update(sp.sympify(loop.stop).atoms(sp.Symbol))

        loop_start_str = str(loop.start).replace('[', '_bracket1_').replace(']', '_bracket2_')
        loop_stop_str = str(loop.stop).replace('[', '_bracket1_').replace(']', '_bracket2_')
        ctr_name = loop.loop_counter_name
        conditions.append(f"{ctr_name} >= {loop_start_str} and {ctr_name} < {loop_stop_str}")

    symbol_names = ','.join([s.name for s in loop_symbols])
    condition_str = ' and '.join(conditions)
    set_description = f"{{ [{symbol_names}] : {condition_str} }}"
    return loop_symbols, isl.BasicSet(set_description)

    for loop in parents_of_type(node, ast.LoopOverCoordinate):
        ctr_name = loop.loop_counter_name
        lower_constraint = isl.Constraint.ineq_from_names(space, {ctr_name: 1, 1: -loop.start})
        upper_constraint = isl.Constraint.ineq_from_names(space, {ctr_name: 1, })


def simplify_conditionals_new(ast_node):
    for conditional in ast_node.atoms(ast.Conditional):
        if conditional.condition_expr == sp.true:
            conditional.parent.replace(conditional, [conditional.true_block])
        elif conditional.condition_expr == sp.false:
            conditional.parent.replace(conditional, [conditional.false_block] if conditional.false_block else [])
        else:
            loop_symbols, iteration_set = isl_iteration_set(conditional)
            symbols_in_condition = conditional.condition_expr.atoms(sp.Symbol)
            if symbols_in_condition.issubset(loop_symbols):
                symbol_names = ','.join([s.name for s in loop_symbols])
                condition_str = str(conditional.condition_expr)
                condition_set = isl.BasicSet(f"{{ [{symbol_names}] : {condition_str} }}")

                intersection = iteration_set.intersect(condition_set)
                if intersection.is_empty():
                    conditional.parent.replace(conditional,
                                               [conditional.false_block] if conditional.false_block else [])
                elif intersection == iteration_set:
                    conditional.parent.replace(conditional, [conditional.true_block])
