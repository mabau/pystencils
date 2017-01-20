import sympy as sp
from pystencils.sympyextensions import replaceAdditive


def sympyCSE(equationCollection):
    """
    Searches for common subexpressions inside the equation collection, in both the existing subexpressions as well
    as the equations themselves. It uses the sympy subexpression detection to do this. Return a new equation collection
    with the additional subexpressions found
    """
    symbolGen = equationCollection.subexpressionSymbolNameGenerator
    replacements, newEq = sp.cse(equationCollection.subexpressions + equationCollection.mainEquations,
                                 symbols=symbolGen)
    replacementEqs = [sp.Eq(*r) for r in replacements]

    modifiedSubexpressions = newEq[:len(equationCollection.subexpressions)]
    modifiedUpdateEquations = newEq[len(equationCollection.subexpressions):]

    newSubexpressions = replacementEqs + modifiedSubexpressions
    topologicallySortedPairs = sp.cse_main.reps_toposort([[e.lhs, e.rhs] for e in newSubexpressions])
    newSubexpressions = [sp.Eq(a[0], a[1]) for a in topologicallySortedPairs]

    return equationCollection.copy(modifiedUpdateEquations, newSubexpressions)


def applyOnAllEquations(equationCollection, operation):
    """Applies sympy expand operation to all equations in collection"""
    result = [operation(s) for s in equationCollection.mainEquations]
    return equationCollection.copy(result)


def applyOnAllSubexpressions(equationCollection, operation):
    return equationCollection.copy(equationCollection.mainEquations,
                                   [operation(s) for s in equationCollection.subexpressions])


def subexpressionSubstitutionInExistingSubexpressions(equationCollection):
    """Goes through the subexpressions list and replaces the term in the following subexpressions"""
    result = []
    for outerCtr, s in enumerate(equationCollection.subexpressions):
        newRhs = s.rhs
        for innerCtr in range(outerCtr):
            subExpr = equationCollection.subexpressions[innerCtr]
            newRhs = replaceAdditive(newRhs, subExpr.lhs, subExpr.rhs, requiredMatchReplacement=1.0)
            newRhs = newRhs.subs(subExpr.rhs, subExpr.lhs)
        result.append(sp.Eq(s.lhs, newRhs))

    return equationCollection.copy(equationCollection.mainEquations, result)


def subexpressionSubstitutionInMainEquations(equationCollection):
    """Replaces already existing subexpressions in the equations of the equationCollection"""
    result = []
    for s in equationCollection.mainEquations:
        newRhs = s.rhs
        for subExpr in equationCollection.subexpressions:
            newRhs = replaceAdditive(newRhs, subExpr.lhs, subExpr.rhs, requiredMatchReplacement=1.0)
        result.append(sp.Eq(s.lhs, newRhs))
    return equationCollection.copy(result)


def addSubexpressionsForDivisions(equationCollection):
    divisors = set()

    def searchDivisors(term):
        if term.func == sp.Pow:
            if term.exp.is_integer and term.exp.is_number and term.exp < 0:
                divisors.add(term)
        else:
            for a in term.args:
                searchDivisors(a)

    for eq in equationCollection.allEquations:
        searchDivisors(eq.rhs)

    newSymbolGen = equationCollection.subexpressionSymbolNameGenerator
    substitutions = {divisor: newSymbol for newSymbol, divisor in zip(newSymbolGen, divisors)}
    return equationCollection.copyWithSubstitutionsApplied(substitutions, True)
