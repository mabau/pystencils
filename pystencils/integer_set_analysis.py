"""Transformations using integer sets based on ISL library"""

import islpy as isl
import sympy as sp

import pystencils.astnodes as ast
from pystencils.typing import parents_of_type
from pystencils.backends.cbackend import CustomSympyPrinter


def remove_brackets(s):
    return s.replace('[', '').replace(']', '')


def _degrees_of_freedom_as_string(expr):
    expr = sp.sympify(expr)
    indexed = expr.atoms(sp.Indexed)
    symbols = expr.atoms(sp.Symbol)
    symbols_without_indexed_base = symbols - {ind.base.args[0] for ind in indexed}
    symbols_without_indexed_base.update(indexed)
    return {remove_brackets(str(s)) for s in symbols_without_indexed_base}


def isl_iteration_set(node: ast.Node):
    """Builds up an ISL set describing the iteration space by analysing the enclosing loops of the given node. """
    conditions = []
    degrees_of_freedom = set()

    for loop in parents_of_type(node, ast.LoopOverCoordinate):
        if loop.step != 1:
            raise NotImplementedError("Loops with strides != 1 are not yet supported.")

        degrees_of_freedom.update(_degrees_of_freedom_as_string(loop.loop_counter_symbol))
        degrees_of_freedom.update(_degrees_of_freedom_as_string(loop.start))
        degrees_of_freedom.update(_degrees_of_freedom_as_string(loop.stop))

        loop_start_str = remove_brackets(str(loop.start))
        loop_stop_str = remove_brackets(str(loop.stop))
        ctr_name = loop.loop_counter_name
        set_string_description = f"{ctr_name} >= {loop_start_str} and {ctr_name} < {loop_stop_str}"
        conditions.append(remove_brackets(set_string_description))

    symbol_names = ','.join(degrees_of_freedom)
    condition_str = ' and '.join(conditions)
    set_description = f"{{ [{symbol_names}] : {condition_str} }}"
    return degrees_of_freedom, isl.BasicSet(set_description)


def simplify_loop_counter_dependent_conditional(conditional):
    """Removes conditionals that depend on the loop counter or iteration limits if they are always true/false."""
    dofs_in_condition = _degrees_of_freedom_as_string(conditional.condition_expr)
    dofs_in_loops, iteration_set = isl_iteration_set(conditional)
    if dofs_in_condition.issubset(dofs_in_loops):
        symbol_names = ','.join(dofs_in_loops)
        condition_str = CustomSympyPrinter().doprint(conditional.condition_expr)
        condition_str = remove_brackets(condition_str)
        condition_set = isl.BasicSet(f"{{ [{symbol_names}] : {condition_str} }}")

        if condition_set.is_empty():
            conditional.replace_by_false_block()
            return

        intersection = iteration_set.intersect(condition_set)
        if intersection.is_empty():
            conditional.replace_by_false_block()
        elif intersection == iteration_set:
            conditional.replace_by_true_block()
