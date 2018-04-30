import sympy as sp
from typing import Callable, List
from pystencils.assignment import Assignment
from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils.sympyextensions import subs_additive

AC = AssignmentCollection


def sympy_cse(ac: AC) -> AC:
    """Searches for common subexpressions inside the equation collection.

    Searches is done in both the existing subexpressions as well as the assignments themselves.
    It uses the sympy subexpression detection to do this. Return a new equation collection
    with the additional subexpressions found
    """
    symbol_gen = ac.subexpression_symbol_generator
    replacements, new_eq = sp.cse(ac.subexpressions + ac.main_assignments,
                                  symbols=symbol_gen)
    replacement_eqs = [Assignment(*r) for r in replacements]

    modified_subexpressions = new_eq[:len(ac.subexpressions)]
    modified_update_equations = new_eq[len(ac.subexpressions):]

    new_subexpressions = replacement_eqs + modified_subexpressions
    topologically_sorted_pairs = sp.cse_main.reps_toposort([[e.lhs, e.rhs] for e in new_subexpressions])
    new_subexpressions = [Assignment(a[0], a[1]) for a in topologically_sorted_pairs]

    return ac.copy(modified_update_equations, new_subexpressions)


def sympy_cse_on_assignment_list(assignments: List[Assignment]) -> List[Assignment]:
    """Extracts common subexpressions from a list of assignments."""
    ec = AC([], assignments)
    return sympy_cse(ec).all_assignments


def subexpression_substitution_in_existing_subexpressions(ac: AC) -> AC:
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


def subexpression_substitution_in_main_assignments(ac: AC) -> AC:
    """Replaces already existing subexpressions in the equations of the assignment_collection."""
    result = []
    for s in ac.main_assignments:
        new_rhs = s.rhs
        for sub_expr in ac.subexpressions:
            new_rhs = subs_additive(new_rhs, sub_expr.lhs, sub_expr.rhs, required_match_replacement=1.0)
        result.append(Assignment(s.lhs, new_rhs))
    return ac.copy(result)


def add_subexpressions_for_divisions(ac: AC) -> AC:
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

    new_symbol_gen = ac.subexpression_symbol_generator
    substitutions = {divisor: new_symbol for new_symbol, divisor in zip(new_symbol_gen, divisors)}
    return ac.new_with_substitutions(substitutions, True)


def apply_to_all_assignments(operation: Callable[[sp.Expr], sp.Expr]) -> Callable[[AC], AC]:
    """Applies sympy expand operation to all equations in collection."""
    def f(assignment_collection: AC) -> AC:
        result = [Assignment(eq.lhs, operation(eq.rhs)) for eq in assignment_collection.main_assignments]
        return assignment_collection.copy(result)
    f.__name__ = operation.__name__
    return f


def apply_on_all_subexpressions(operation: Callable[[sp.Expr], sp.Expr]) -> Callable[[AC], AC]:
    """Applies the given operation on all subexpressions of the AC."""
    def f(ac: AC) -> AC:
        result = [Assignment(eq.lhs, operation(eq.rhs)) for eq in ac.subexpressions]
        return ac.copy(ac.main_assignments, result)
    f.__name__ = operation.__name__
    return f