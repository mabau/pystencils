import sympy as sp

from pystencils import Assignment, AssignmentCollection
from pystencils.sympyextensions import replaceAdditive


def sympyCseOnEquationList(eqs):
    ec = AssignmentCollection(eqs, [])
    return sympyCSE(ec).allEquations


def sympyCSE(assignment_collection):
    """
    Searches for common subexpressions inside the equation collection, in both the existing subexpressions as well
    as the equations themselves. It uses the sympy subexpression detection to do this. Return a new equation collection
    with the additional subexpressions found
    """
    symbolGen = assignment_collection.subexpressionSymbolNameGenerator
    replacements, newEq = sp.cse(assignment_collection.subexpressions + assignment_collection.mainAssignments,
                                 symbols=symbolGen)
    replacementEqs = [Assignment(*r) for r in replacements]

    modifiedSubexpressions = newEq[:len(assignment_collection.subexpressions)]
    modifiedUpdateEquations = newEq[len(assignment_collection.subexpressions):]

    newSubexpressions = replacementEqs + modifiedSubexpressions
    topologicallySortedPairs = sp.cse_main.reps_toposort([[e.lhs, e.rhs] for e in newSubexpressions])
    newSubexpressions = [Assignment(a[0], a[1]) for a in topologicallySortedPairs]

    return assignment_collection.copy(modifiedUpdateEquations, newSubexpressions)


def applyOnAllEquations(assignment_collection, operation):
    """Applies sympy expand operation to all equations in collection"""
    result = [Assignment(eq.lhs, operation(eq.rhs)) for eq in assignment_collection.mainAssignments]
    return assignment_collection.copy(result)


def applyOnAllSubexpressions(assignment_collection, operation):
    result = [Assignment(eq.lhs, operation(eq.rhs)) for eq in assignment_collection.subexpressions]
    return assignment_collection.copy(assignment_collection.mainAssignments, result)


def subexpressionSubstitutionInExistingSubexpressions(assignment_collection):
    """Goes through the subexpressions list and replaces the term in the following subexpressions"""
    result = []
    for outerCtr, s in enumerate(assignment_collection.subexpressions):
        newRhs = s.rhs
        for innerCtr in range(outerCtr):
            subExpr = assignment_collection.subexpressions[innerCtr]
            newRhs = replaceAdditive(newRhs, subExpr.lhs, subExpr.rhs, requiredMatchReplacement=1.0)
            newRhs = newRhs.subs(subExpr.rhs, subExpr.lhs)
        result.append(Assignment(s.lhs, newRhs))

    return assignment_collection.copy(assignment_collection.mainAssignments, result)


def subexpressionSubstitutionInmainAssignments(assignment_collection):
    """Replaces already existing subexpressions in the equations of the assignment_collection"""
    result = []
    for s in assignment_collection.mainAssignments:
        newRhs = s.rhs
        for subExpr in assignment_collection.subexpressions:
            newRhs = replaceAdditive(newRhs, subExpr.lhs, subExpr.rhs, requiredMatchReplacement=1.0)
        result.append(Assignment(s.lhs, newRhs))
    return assignment_collection.copy(result)


def addSubexpressionsForDivisions(assignment_collection):
    """Introduces subexpressions for all divisions which have no constant in the denominator.
    e.g.  :math:`\frac{1}{x}` is replaced, :math:`\frac{1}{3}` is not replaced."""
    divisors = set()

    def searchDivisors(term):
        if term.func == sp.Pow:
            if term.exp.is_integer and term.exp.is_number and term.exp < 0:
                divisors.add(term)
        else:
            for a in term.args:
                searchDivisors(a)

    for eq in assignment_collection.allEquations:
        searchDivisors(eq.rhs)

    newSymbolGen = assignment_collection.subexpressionSymbolNameGenerator
    substitutions = {divisor: newSymbol for newSymbol, divisor in zip(newSymbolGen, divisors)}
    return assignment_collection.copyWithSubstitutionsApplied(substitutions, True)
