import sympy as sp
from pystencils.transformations import fastSubs


class EquationCollection:
    """
    A collection of equations with subexpression definitions, also represented as equations,
    that are used in the main equations. EquationCollections can be passed to simplification methods.
    These simplification methods can change the subexpressions, but the number and
    left hand side of the main equations themselves is not altered.
    Additionally a dictionary of simplification hints is stored, which are set by the functions that create
    equation collections to transport information to the simplification system.

    :ivar mainEquations: list of sympy equations
    :ivar subexpressions: list of sympy equations defining subexpressions used in main equations
    :ivar simplificationHints: dictionary that is used to annotate the equation collection with hints that are
                               used by the simplification system. See documentation of the simplification rules for
                               potentially required hints and their meaning.
    """

    # ----------------------------------------- Creation ---------------------------------------------------------------

    def __init__(self, equations, subExpressions, simplificationHints={}):
        self.mainEquations = equations
        self.subexpressions = subExpressions
        self.simplificationHints = simplificationHints

        def symbolGen():
            """Use this generator to create new unused symbols for subexpressions"""
            counter = 0
            while True:
                counter += 1
                newSymbol = sp.Symbol("xi_" + str(counter))
                if newSymbol in self.boundSymbols:
                    continue
                yield newSymbol

        self.subexpressionSymbolNameGenerator = symbolGen()

    def createNewWithAdditionalSubexpressions(self, newEquations, additionalSubExpressions):
        assert len(self.mainEquations) == len(newEquations), "Number of update equations cannot be changed"
        return EquationCollection(newEquations,
                                  self.subexpressions + additionalSubExpressions,
                                  self.simplificationHints)

    def createNewWithSubstitutionsApplied(self, substitutionDict):
        newSubexpressions = [fastSubs(eq, substitutionDict) for eq in self.subexpressions]
        newEquations = [fastSubs(eq, substitutionDict) for eq in self.mainEquations]
        return EquationCollection(newEquations, newSubexpressions, self.simplificationHints)

    def addSimplificationHint(self, key, value):
        assert key not in self.simplificationHints, "This hint already exists"
        self.simplificationHints[key] = value

    # ---------------------------------------------- Properties  -------------------------------------------------------

    @property
    def allEquations(self):
        return self.subexpressions + self.mainEquations

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
        assert len(boundSymbolsSet) == len(self.subexpressions) + len(self.mainEquations), \
            "Not in SSA form - same symbol assigned multiple times"
        return boundSymbolsSet

    @property
    def definedSymbols(self):
        """All symbols that occur as left-hand-sides of the main equations"""
        return set([eq.lhs for eq in self.mainEquations])

    # ----------------------------------------- Display and Printing   -------------------------------------------------

    def _repr_html_(self):
        def makeHtmlEquationTable(equations):
            noBorder = 'style="border:none"'
            htmlTable = '<table style="border:none; width: 100%; ">'
            line = '<tr {nb}> <td {nb}>${lhs}$</td> <td {nb}>$=$</td> ' \
                   '<td style="border:none; width: 100%;">${rhs}$</td> </tr>'
            for eq in equations:
                formatDict = {'lhs': sp.latex(eq.lhs),
                              'rhs': sp.latex(eq.rhs),
                              'nb': noBorder, }
                htmlTable += line.format(**formatDict)
            htmlTable += "</table>"
            return htmlTable

        result = ""
        if len(self.subexpressions) > 0:
            result += "<div>Subexpressions:<div>"
            result += makeHtmlEquationTable(self.subexpressions)
        result += "<div>Main Equations:<div>"
        result += makeHtmlEquationTable(self.mainEquations)
        return result

    def __repr__(self):
        return "Equation Collection for " + ",".join([str(eq.lhs) for eq in self.mainEquations])

    # -------------------------------------   Manipulation  ------------------------------------------------------------

    def merge(self, other):
        """Returns a new collection which contains self and other. Subexpressions are renamed if they clash."""
        ownDefs = set([e.lhs for e in self.mainEquations])
        otherDefs = set([e.lhs for e in other.mainEquations])
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
                    newLhs = self.subexpressionSymbolNameGenerator()
                    newEq = sp.Eq(newLhs, fastSubs(otherSubexpressionEq.rhs, substitutionDict))
                    processedOtherSubexpressionEquations.append(newEq)
                    substitutionDict[otherSubexpressionEq.lhs] = newLhs
            else:
                processedOtherSubexpressionEquations.append(fastSubs(otherSubexpressionEq, substitutionDict))
        return EquationCollection(self.mainEquations + other.mainEquations,
                                  self.subexpressions + processedOtherSubexpressionEquations)

    def extract(self, symbolsToExtract):
        """
        Creates a new equation collection with equations that have symbolsToExtract as left-hand-sides and
        only the necessary subexpressions that are used in these equations
        """
        symbolsToExtract = set(symbolsToExtract)
        newEquations = []

        subexprMap = {e.lhs: e.rhs for e in self.subexpressions}
        handledSymbols = set()
        queue = []

        def addSymbolsFromExpr(expr):
            dependentSymbols = expr.atoms(sp.Symbol)
            for ds in dependentSymbols:
                if ds not in handledSymbols:
                    queue.append(ds)
                    handledSymbols.add(ds)

        for eq in self.mainEquations:
            if eq.lhs in symbolsToExtract:
                newEquations.append(eq)
                addSymbolsFromExpr(eq.rhs)

        while len(queue) > 0:
            e = queue.pop(0)
            if e not in subexprMap:
                continue
            else:
                addSymbolsFromExpr(subexprMap[e])

        newSubExpr = [eq for eq in self.subexpressions if eq.lhs in handledSymbols]
        return EquationCollection(newEquations, newSubExpr)

    def newWithoutUnusedSubexpressions(self):
        """Returns a new equation collection containing only the subexpressions that
        are used/referenced in the equations"""
        allLhs = [eq.lhs for eq in self.mainEquations]
        return self.extract(allLhs)

    def insertSubexpressions(self):
        """Returns a new equation collection by inserting all subexpressions into the main equations"""
        if len(self.subexpressions) == 0:
            return EquationCollection(self.mainEquations, self.subexpressions, self.simplificationHints)
        subsDict = {self.subexpressions[0].lhs: self.subexpressions[0].rhs}
        subExpr = [e for e in self.subexpressions]
        for i in range(1, len(subExpr)):
            subExpr[i] = fastSubs(subExpr[i], subsDict)
            subsDict[subExpr[i].lhs] = subExpr[i].rhs

        newEq = [fastSubs(eq, subsDict) for eq in self.mainEquations]
        return EquationCollection(newEq, [], self.simplificationHints)

    def lambdify(self, symbols, module=None, fixedSymbols={}):
        """
        Returns a function to evaluate this equation collection
        :param symbols: symbol(s) which are the parameter for the created function
        :param module: same as sympy.lambdify paramter of same same, i.e. which module to use e.g. 'numpy'
        :param fixedSymbols: dictionary with substitutions, that are applied before lambdification
        """
        eqs = self.createNewWithSubstitutionsApplied(fixedSymbols).insertSubexpressions().mainEquations
        print('abc')
        for eq in eqs:
            print(eq)
            sp.lambdify(eq.rhs, symbols, module)
        lambdas = {eq.lhs: sp.lambdify(eq.rhs, symbols, module) for eq in eqs}

        def f(*args, **kwargs):
            return {s: f(*args, **kwargs) for s, f in lambdas.items()}

        return f
