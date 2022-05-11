from itertools import chain
from typing import Callable, List, Sequence, Union
from collections import defaultdict

import sympy as sp

from pystencils.assignment import Assignment
from pystencils.astnodes import Node
from pystencils.field import Field
from pystencils.sympyextensions import subs_additive, is_constant, recursive_collect


def sort_assignments_topologically(assignments: Sequence[Union[Assignment, Node]]) -> List[Union[Assignment, Node]]:
    """Sorts assignments in topological order, such that symbols used on rhs occur first on a lhs"""
    edges = []
    for c1, e1 in enumerate(assignments):
        if hasattr(e1, 'lhs') and hasattr(e1, 'rhs'):
            symbols = [e1.lhs]
        elif isinstance(e1, Node):
            symbols = e1.symbols_defined
        else:
            raise NotImplementedError(f"Cannot sort topologically. Object of type {type(e1)} cannot be handled.")

        for lhs in symbols:
            for c2, e2 in enumerate(assignments):
                if isinstance(e2, Assignment) and lhs in e2.rhs.free_symbols:
                    edges.append((c1, c2))
                elif isinstance(e2, Node) and lhs in e2.undefined_symbols:
                    edges.append((c1, c2))
    return [assignments[i] for i in sp.topological_sort((range(len(assignments)), edges))]


def sympy_cse(ac, **kwargs):
    """Searches for common subexpressions inside the assignment collection.

    Searches is done in both the existing subexpressions as well as the assignments themselves.
    It uses the sympy subexpression detection to do this. Return a new assignment collection
    with the additional subexpressions found
    """
    symbol_gen = ac.subexpression_symbol_generator

    all_assignments = [e for e in chain(ac.subexpressions, ac.main_assignments) if isinstance(e, Assignment)]
    other_objects = [e for e in chain(ac.subexpressions, ac.main_assignments) if not isinstance(e, Assignment)]
    replacements, new_eq = sp.cse(all_assignments, symbols=symbol_gen, **kwargs)

    replacement_eqs = [Assignment(*r) for r in replacements]

    modified_subexpressions = new_eq[:len(ac.subexpressions)]
    modified_update_equations = new_eq[len(ac.subexpressions):]

    new_subexpressions = sort_assignments_topologically(other_objects + replacement_eqs + modified_subexpressions)
    return ac.copy(modified_update_equations, new_subexpressions)


def sympy_cse_on_assignment_list(assignments: List[Assignment]) -> List[Assignment]:
    """Extracts common subexpressions from a list of assignments."""
    from pystencils.simp.assignment_collection import AssignmentCollection
    ec = AssignmentCollection([], assignments)
    return sympy_cse(ec).all_assignments


def subexpression_substitution_in_existing_subexpressions(ac):
    """Goes through the subexpressions list and replaces the term in the following subexpressions."""
    result = []
    for outer_ctr, s in enumerate(ac.subexpressions):
        new_rhs = s.rhs
        for inner_ctr in range(outer_ctr):
            sub_expr = ac.subexpressions[inner_ctr]
            new_rhs = subs_additive(new_rhs, sub_expr.lhs, sub_expr.rhs, required_match_replacement=1.0)
            new_rhs = new_rhs.subs(sub_expr.rhs, sub_expr.lhs)
        result.append(Assignment(s.lhs, new_rhs))

    return ac.copy(ac.main_assignments, result)


def subexpression_substitution_in_main_assignments(ac):
    """Replaces already existing subexpressions in the equations of the assignment_collection."""
    result = []
    for s in ac.main_assignments:
        new_rhs = s.rhs
        for sub_expr in ac.subexpressions:
            new_rhs = subs_additive(new_rhs, sub_expr.lhs, sub_expr.rhs, required_match_replacement=1.0)
        result.append(Assignment(s.lhs, new_rhs))
    return ac.copy(result)


def add_subexpressions_for_constants(ac):
    """Extracts constant factors to subexpressions in the given assignment collection.

    SymPy will exclude common factors from a sum only if they are symbols. This simplification
    can be applied to exclude common numeric constants from multiple terms of a sum. As a consequence,
    the number of multiplications is reduced and in some cases, more common subexpressions can be found.
    """
    constants_to_subexp_dict = defaultdict(lambda: next(ac.subexpression_symbol_generator))

    def visit(expr):
        args = list(expr.args)
        if len(args) == 0:
            return expr
        if isinstance(expr, sp.Add) or isinstance(expr, sp.Mul):
            for i, arg in enumerate(args):
                if is_constant(arg) and abs(arg) != 1:
                    if arg < 0:
                        args[i] = - constants_to_subexp_dict[- arg]
                    else:
                        args[i] = constants_to_subexp_dict[arg]
        return expr.func(*(visit(a) for a in args))
    main_assignments = [Assignment(a.lhs, visit(a.rhs)) for a in ac.main_assignments]
    subexpressions = [Assignment(a.lhs, visit(a.rhs)) for a in ac.subexpressions]

    symbols_to_collect = set(constants_to_subexp_dict.values())

    main_assignments = [Assignment(a.lhs, recursive_collect(a.rhs, symbols_to_collect, True)) for a in main_assignments]
    subexpressions = [Assignment(a.lhs, recursive_collect(a.rhs, symbols_to_collect, True)) for a in subexpressions]

    subexpressions = [Assignment(symb, c) for c, symb in constants_to_subexp_dict.items()] + subexpressions
    return ac.copy(main_assignments=main_assignments, subexpressions=subexpressions)


def add_subexpressions_for_divisions(ac):
    r"""Introduces subexpressions for all divisions which have no constant in the denominator.

    For example :math:`\frac{1}{x}` is replaced while :math:`\frac{1}{3}` is not replaced.
    """
    divisors = set()

    def search_divisors(term):
        if term.func == sp.Pow:
            if term.exp.is_integer and term.exp.is_number and term.exp < 0:
                divisors.add(term)
        else:
            for a in term.args:
                search_divisors(a)

    for eq in ac.all_assignments:
        search_divisors(eq.rhs)

    divisors = sorted(list(divisors), key=lambda x: str(x))
    new_symbol_gen = ac.subexpression_symbol_generator
    substitutions = {divisor: new_symbol for new_symbol, divisor in zip(new_symbol_gen, divisors)}
    return ac.new_with_substitutions(substitutions, add_substitutions_as_subexpressions=True, substitute_on_lhs=False)


def add_subexpressions_for_sums(ac):
    r"""Introduces subexpressions for all sums - i.e. splits addends into subexpressions."""
    addends = []

    def contains_sum(term):
        if term.func == sp.Add:
            return True
        if term.is_Atom:
            return False
        return any([contains_sum(a) for a in term.args])

    def search_addends(term):
        if term.func == sp.Add:
            if all([not contains_sum(a) for a in term.args]):
                addends.extend(term.args)
        for a in term.args:
            search_addends(a)

    for eq in ac.all_assignments:
        search_addends(eq.rhs)

    addends = [a for a in addends if not isinstance(a, sp.Symbol) or isinstance(a, Field.Access)]
    new_symbol_gen = ac.subexpression_symbol_generator
    substitutions = {addend: new_symbol for new_symbol, addend in zip(new_symbol_gen, addends)}
    return ac.new_with_substitutions(substitutions, True, substitute_on_lhs=False)


def add_subexpressions_for_field_reads(ac, subexpressions=True, main_assignments=True):
    r"""Substitutes field accesses on rhs of assignments with subexpressions

    Can change semantics of the update rule (which is the goal of this transformation)
    This is useful if a field should be update in place - all values are loaded before into subexpression variables,
    then the new values are computed and written to the same field in-place.
    """
    field_reads = set()
    to_iterate = []
    if subexpressions:
        to_iterate = chain(to_iterate, ac.subexpressions)
    if main_assignments:
        to_iterate = chain(to_iterate, ac.main_assignments)

    for assignment in to_iterate:
        if hasattr(assignment, 'lhs') and hasattr(assignment, 'rhs'):
            field_reads.update(assignment.rhs.atoms(Field.Access))
    substitutions = {fa: next(ac.subexpression_symbol_generator) for fa in field_reads}
    return ac.new_with_substitutions(substitutions, add_substitutions_as_subexpressions=True,
                                     substitute_on_lhs=False, sort_topologically=False)


def transform_rhs(assignment_list, transformation, *args, **kwargs):
    """Applies a transformation function on the rhs of each element of the passed assignment list
    If the list also contains other object, like AST nodes, these are ignored.
    Additional parameters are passed to the transformation function"""
    return [Assignment(a.lhs, transformation(a.rhs, *args, **kwargs)) if hasattr(a, 'lhs') and hasattr(a, 'rhs') else a
            for a in assignment_list]


def transform_lhs_and_rhs(assignment_list, transformation, *args, **kwargs):
    return [Assignment(transformation(a.lhs, *args, **kwargs),
                       transformation(a.rhs, *args, **kwargs))
            if hasattr(a, 'lhs') and hasattr(a, 'rhs') else a
            for a in assignment_list]


def apply_to_all_assignments(operation: Callable[[sp.Expr], sp.Expr]):
    """Applies a given operation to all equations in collection."""

    def f(ac):
        return ac.copy(transform_rhs(ac.main_assignments, operation))

    f.__name__ = operation.__name__
    return f


def apply_on_all_subexpressions(operation: Callable[[sp.Expr], sp.Expr]):
    """Applies the given operation on all subexpressions of the AC."""

    def f(ac):
        return ac.copy(ac.main_assignments, transform_rhs(ac.subexpressions, operation))

    f.__name__ = operation.__name__
    return f

# TODO Markus
# make this really work for Assignmentcollections
# this function should ONLY evaluate
# do the optims_c99 elsewhere optionally

# def apply_sympy_optimisations(ac: AssignmentCollection):
#     """ Evaluates constant expressions (e.g. :math:`\\sqrt{3}` will be replaced by its floating point representation)
#         and applies the default sympy optimisations. See sympy.codegen.rewriting
#     """
#
#     # Evaluates all constant terms
#
#     assignments = ac.all_assignments
#
#     evaluate_constant_terms = ReplaceOptim(lambda e: hasattr(e, 'is_constant') and e.is_constant and not e.is_integer,
#                                            lambda p: p.evalf())
#
#     sympy_optimisations = [evaluate_constant_terms] + list(optims_c99)
#
#     assignments = [Assignment(a.lhs, optimize(a.rhs, sympy_optimisations))
#                    if hasattr(a, 'lhs')
#                    else a for a in assignments]
#     assignments_nodes = [a.atoms(SympyAssignment) for a in assignments]
#     for a in chain.from_iterable(assignments_nodes):
#         a.optimize(sympy_optimisations)
#
#     return AssignmentCollection(assignments)
