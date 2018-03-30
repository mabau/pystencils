import sympy as sp
from copy import copy
from pystencils.assignment import Assignment
from pystencils.sympyextensions import fastSubs, countNumberOfOperations, sortEquationsTopologically


class AssignmentCollection(object):
    """
    A collection of equations with subexpression definitions, also represented as equations,
    that are used in the main equations. AssignmentCollection can be passed to simplification methods.
    These simplification methods can change the subexpressions, but the number and
    left hand side of the main equations themselves is not altered.
    Additionally a dictionary of simplification hints is stored, which are set by the functions that create
    equation collections to transport information to the simplification system.

    :ivar mainAssignments: list of sympy equations
    :ivar subexpressions: list of sympy equations defining subexpressions used in main equations
    :ivar simplificationHints: dictionary that is used to annotate the equation collection with hints that are
                               used by the simplification system. See documentation of the simplification rules for
                               potentially required hints and their meaning.
    """

    # ----------------------------------------- Creation ---------------------------------------------------------------

    def __init__(self, equations, subExpressions, simplificationHints=None, subexpressionSymbolNameGenerator=None):
        self.mainAssignments = equations
        self.subexpressions = subExpressions

        if simplificationHints is None:
            simplificationHints = {}

        self.simplificationHints = simplificationHints

        if subexpressionSymbolNameGenerator is None:
            self.subexpressionSymbolNameGenerator = SymbolGen()
        else:
            self.subexpressionSymbolNameGenerator = subexpressionSymbolNameGenerator

    @property
    def mainTerms(self):
        return []

    def copy(self, mainAssignments=None, subexpressions=None):
        res = copy(self)
        res.simplificationHints = self.simplificationHints.copy()
        res.subexpressionSymbolNameGenerator = copy(self.subexpressionSymbolNameGenerator)

        if mainAssignments is not None:
            res.mainAssignments = mainAssignments
        else:
            res.mainAssignments = self.mainAssignments.copy()

        if subexpressions is not None:
            res.subexpressions = subexpressions
        else:
            res.subexpressions = self.subexpressions.copy()

        return res

    def copyWithSubstitutionsApplied(self, substitutionDict, addSubstitutionsAsSubexpressions=False,
                                     substituteOnLhs=True):
        """
        Returns a new equation collection, where terms are substituted according to the passed `substitutionDict`.
        Substitutions are made in the subexpression terms and the main equations
        """
        if substituteOnLhs:
            newSubexpressions = [fastSubs(eq, substitutionDict) for eq in self.subexpressions]
            newEquations = [fastSubs(eq, substitutionDict) for eq in self.mainAssignments]
        else:
            newSubexpressions = [Assignment(eq.lhs, fastSubs(eq.rhs, substitutionDict)) for eq in self.subexpressions]
            newEquations = [Assignment(eq.lhs, fastSubs(eq.rhs, substitutionDict)) for eq in self.mainAssignments]

        if addSubstitutionsAsSubexpressions:
            newSubexpressions = [Assignment(b, a) for a, b in substitutionDict.items()] + newSubexpressions
            newSubexpressions = sortEquationsTopologically(newSubexpressions)
        return self.copy(newEquations, newSubexpressions)

    def addSimplificationHint(self, key, value):
        """
        Adds an entry to the simplificationHints dictionary, and checks that is does not exist yet
        """
        assert key not in self.simplificationHints, "This hint already exists"
        self.simplificationHints[key] = value

    # ---------------------------------------------- Properties  -------------------------------------------------------

    @property
    def allEquations(self):
        """Subexpression and main equations in one sequence"""
        return self.subexpressions + self.mainAssignments

    @property
    def freeSymbols(self):
        """All symbols used in the equation collection, which have not been defined inside the equation system"""
        freeSymbols = set()
        for eq in self.allEquations:
            freeSymbols.update(eq.rhs.atoms(sp.Symbol))
        return freeSymbols - self.boundSymbols

    @property
    def boundSymbols(self):
        """Set of all symbols which occur on left-hand-sides i.e. all symbols which are defined."""
        boundSymbolsSet = set([eq.lhs for eq in self.allEquations])
        assert len(boundSymbolsSet) == len(self.subexpressions) + len(self.mainAssignments), \
            "Not in SSA form - same symbol assigned multiple times"
        return boundSymbolsSet

    @property
    def definedSymbols(self):
        """All symbols that occur as left-hand-sides of the main equations"""
        return set([eq.lhs for eq in self.mainAssignments])

    @property
    def operationCount(self):
        """See :func:`countNumberOfOperations` """
        return countNumberOfOperations(self.allEquations, onlyType=None)

    def get(self, symbols, frommainAssignmentsOnly=False):
        """Return the equations which have symbols as left hand sides"""
        if not hasattr(symbols, "__len__"):
            symbols = list(symbols)
        symbols = set(symbols)

        if not frommainAssignmentsOnly:
            eqsToSearchIn = self.allEquations
        else:
            eqsToSearchIn = self.mainAssignments

        return [eq for eq in eqsToSearchIn if eq.lhs in symbols]

    # ----------------------------------------- Display and Printing   -------------------------------------------------

    def _repr_html_(self):
        def makeHtmlEquationTable(equations):
            noBorder = 'style="border:none"'
            htmlTable = '<table style="border:none; width: 100%; ">'
            line = '<tr {nb}> <td {nb}>$${eq}$$</td>  </tr> '
            for eq in equations:
                formatDict = {'eq': sp.latex(eq),
                              'nb': noBorder, }
                htmlTable += line.format(**formatDict)
            htmlTable += "</table>"
            return htmlTable

        result = ""
        if len(self.subexpressions) > 0:
            result += "<div>Subexpressions:</div>"
            result += makeHtmlEquationTable(self.subexpressions)
        result += "<div>Main Assignments:</div>"
        result += makeHtmlEquationTable(self.mainAssignments)
        return result

    def __repr__(self):
        return "Equation Collection for " + ",".join([str(eq.lhs) for eq in self.mainAssignments])

    def __str__(self):
        result = "Subexpressions\n"
        for eq in self.subexpressions:
            result += str(eq) + "\n"
        result += "Main Assignments\n"
        for eq in self.mainAssignments:
            result += str(eq) + "\n"
        return result

    # -------------------------------------   Manipulation  ------------------------------------------------------------

    def merge(self, other):
        """Returns a new collection which contains self and other. Subexpressions are renamed if they clash."""
        ownDefs = set([e.lhs for e in self.mainAssignments])
        otherDefs = set([e.lhs for e in other.mainAssignments])
        assert len(ownDefs.intersection(otherDefs)) == 0, "Cannot merge, since both collection define the same symbols"

        ownSubexpressionSymbols = {e.lhs: e.rhs for e in self.subexpressions}
        substitutionDict = {}

        processedOtherSubexpressionEquations = []
        for otherSubexpressionEq in other.subexpressions:
            if otherSubexpressionEq.lhs in ownSubexpressionSymbols:
                if otherSubexpressionEq.rhs == ownSubexpressionSymbols[otherSubexpressionEq.lhs]:
                    continue  # exact the same subexpression equation exists already
                else:
                    # different definition - a new name has to be introduced
                    newLhs = next(self.subexpressionSymbolNameGenerator)
                    newEq = Assignment(newLhs, fastSubs(otherSubexpressionEq.rhs, substitutionDict))
                    processedOtherSubexpressionEquations.append(newEq)
                    substitutionDict[otherSubexpressionEq.lhs] = newLhs
            else:
                processedOtherSubexpressionEquations.append(fastSubs(otherSubexpressionEq, substitutionDict))

        processedOthermainAssignments = [fastSubs(eq, substitutionDict) for eq in other.mainAssignments]
        return self.copy(self.mainAssignments + processedOthermainAssignments,
                         self.subexpressions + processedOtherSubexpressionEquations)

    def getDependentSymbols(self, symbolSequence):
        """Returns a list of symbols that depend on the passed symbols."""

        queue = list(symbolSequence)

        def addSymbolsFromExpr(expr):
            dependentSymbols = expr.atoms(sp.Symbol)
            for ds in dependentSymbols:
                queue.append(ds)

        handledSymbols = set()
        eqMap = {e.lhs: e.rhs for e in self.allEquations}

        while len(queue) > 0:
            e = queue.pop(0)
            if e in handledSymbols:
                continue
            if e in eqMap:
                addSymbolsFromExpr(eqMap[e])
            handledSymbols.add(e)

        return handledSymbols

    def extract(self, symbolsToExtract):
        """
        Creates a new equation collection with equations that have symbolsToExtract as left-hand-sides and
        only the necessary subexpressions that are used in these equations
        """
        symbolsToExtract = set(symbolsToExtract)
        dependentSymbols = self.getDependentSymbols(symbolsToExtract)
        newEquations = []
        for eq in self.allEquations:
            if eq.lhs in symbolsToExtract:
                newEquations.append(eq)

        newSubExpr = [eq for eq in self.subexpressions if eq.lhs in dependentSymbols and eq.lhs not in symbolsToExtract]
        return AssignmentCollection(newEquations, newSubExpr)

    def newWithoutUnusedSubexpressions(self):
        """Returns a new equation collection containing only the subexpressions that
        are used/referenced in the equations"""
        allLhs = [eq.lhs for eq in self.mainAssignments]
        return self.extract(allLhs)

    def appendToSubexpressions(self, rhs, lhs=None, topologicalSort=True):
        if lhs is None:
            lhs = sp.Dummy()
        eq = Assignment(lhs, rhs)
        self.subexpressions.append(eq)
        if topologicalSort:
            self.topologicalSort(subexpressions=True, mainAssignments=False)
        return lhs

    def topologicalSort(self, subexpressions=True, mainAssignments=True):
        if subexpressions:
            self.subexpressions = sortEquationsTopologically(self.subexpressions)
        if mainAssignments:
            self.mainAssignments = sortEquationsTopologically(self.mainAssignments)

    def insertSubexpression(self, symbol):
        newSubexpressions = []
        subsDict = None
        for se in self.subexpressions:
            if se.lhs == symbol:
                subsDict = {se.lhs: se.rhs}
            else:
                newSubexpressions.append(se)
        if subsDict is None:
            return self

        newSubexpressions = [Assignment(eq.lhs, fastSubs(eq.rhs, subsDict)) for eq in newSubexpressions]
        newEqs = [Assignment(eq.lhs, fastSubs(eq.rhs, subsDict)) for eq in self.mainAssignments]
        return self.copy(newEqs, newSubexpressions)

    def insertSubexpressions(self, subexpressionSymbolsToKeep=set()):
        """Returns a new equation collection by inserting all subexpressions into the main equations"""
        if len(self.subexpressions) == 0:
            return self.copy()

        subexpressionSymbolsToKeep = set(subexpressionSymbolsToKeep)

        keptSubexpressions = []
        if self.subexpressions[0].lhs in subexpressionSymbolsToKeep:
            subsDict = {}
            keptSubexpressions = self.subexpressions[0]
        else:
            subsDict = {self.subexpressions[0].lhs: self.subexpressions[0].rhs}

        subExpr = [e for e in self.subexpressions]
        for i in range(1, len(subExpr)):
            subExpr[i] = fastSubs(subExpr[i], subsDict)
            if subExpr[i].lhs in subexpressionSymbolsToKeep:
                keptSubexpressions.append(subExpr[i])
            else:
                subsDict[subExpr[i].lhs] = subExpr[i].rhs

        newEq = [fastSubs(eq, subsDict) for eq in self.mainAssignments]
        return self.copy(newEq, keptSubexpressions)

    def lambdify(self, symbols, module=None, fixedSymbols={}):
        """
        Returns a function to evaluate this equation collection
        :param symbols: symbol(s) which are the parameter for the created function
        :param module: same as sympy.lambdify paramter of same same, i.e. which module to use e.g. 'numpy'
        :param fixedSymbols: dictionary with substitutions, that are applied before lambdification
        """
        eqs = self.copyWithSubstitutionsApplied(fixedSymbols).insertSubexpressions().mainAssignments
        lambdas = {eq.lhs: sp.lambdify(symbols, eq.rhs, module) for eq in eqs}

        def f(*args, **kwargs):
            return {s: f(*args, **kwargs) for s, f in lambdas.items()}

        return f


class SymbolGen:
    def __init__(self):
        self._ctr = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._ctr += 1
        return sp.Symbol("xi_" + str(self._ctr))

    def next(self):
        return self.__next__()