import sympy as sp
from collections import namedtuple, defaultdict
from pystencils.sympyextensions import normalizeProduct, prod


def defaultDiffSortKey(d):
    return str(d.superscript), str(d.target)


class Diff(sp.Expr):
    """
    Sympy Node representing a derivative. The difference to sympy's built in differential is:
        - shortened latex representation
        - all simplifications have to be done manually
        - optional marker displayed as superscript
    """
    is_number = False
    is_Rational = False

    def __new__(cls, argument, target=-1, superscript=-1, **kwargs):
        if argument == 0:
            return sp.Rational(0, 1)
        return sp.Expr.__new__(cls, argument.expand(), sp.sympify(target), sp.sympify(superscript), **kwargs)

    @property
    def is_commutative(self):
        anyNonCommutative = any(not s.is_commutative for s in self.atoms(sp.Symbol))
        if anyNonCommutative:
            return False
        else:
            return True

    def getArgRecursive(self):
        """Returns the argument the derivative acts on, for nested derivatives the inner argument is returned"""
        if not isinstance(self.arg, Diff):
            return self.arg
        else:
            return self.arg.getArgRecursive()

    def changeArgRecursive(self, newArg):
        """Returns a Diff node with the given 'newArg' instead of the current argument. For nested derivatives
        a new nested derivative is returned where the inner Diff has the 'newArg'"""
        if not isinstance(self.arg, Diff):
            return Diff(newArg, self.target, self.superscript)
        else:
            return Diff(self.arg.changeArgRecursive(newArg), self.target, self.superscript)

    def splitLinear(self, functions):
        """
        Applies linearity property of Diff: i.e.  'Diff(c*a+b)' is transformed to 'c * Diff(a) + Diff(b)'
        The parameter functions is a list of all symbols that are considered functions, not constants.
        For the example above: functions=[a, b]
        """
        constant, variable = 1, 1

        if self.arg.func != sp.Mul:
            constant, variable = 1, self.arg
        else:
            for factor in normalizeProduct(self.arg):
                if factor in functions or isinstance(factor, Diff):
                    variable *= factor
                else:
                    constant *= factor

        if isinstance(variable, sp.Symbol) and variable not in functions:
            return 0

        if isinstance(variable, int) or variable.is_number:
            return 0
        else:
            return constant * Diff(variable, target=self.target, superscript=self.superscript)

    @property
    def arg(self):
        """Expression the derivative acts on"""
        return self.args[0]

    @property
    def target(self):
        """Subscript, usually the variable the Diff is w.r.t. """
        return self.args[1]

    @property
    def superscript(self):
        """Superscript, used as the Chapman Enskog order index"""
        return self.args[2]

    def _latex(self, printer, *args):
        result = "{\partial"
        if self.superscript >= 0:
            result += "^{(%s)}" % (self.superscript,)
        if self.target != -1:
            result += "_{%s}" % (printer.doprint(self.target),)

        contents = printer.doprint(self.arg)
        if isinstance(self.arg, int) or isinstance(self.arg, sp.Symbol) or self.arg.is_number or self.arg.func == Diff:
            result += " " + contents
        else:
            result += " (" + contents + ") "

        result += "}"
        return result

    def __str__(self):
        return "D(%s)" % self.arg


class DiffOperator(sp.Expr):
    """
    Un-applied differential, i.e. differential operator
    Its args are:
        - target: the differential is w.r.t to this variable.
                 This target is mainly for display purposes (its the subscript) and to distinguish DiffOperators
                 If the target is '-1' no subscript is displayed
        - superscript: optional marker displayed as superscript
                        is not displayed if set to '-1'
    The DiffOperator behaves much like a variable with special name. Its main use is to be applied later, using the
    DiffOperator.apply(expr, arg) which transforms 'DiffOperator's to applied 'Diff's
    """
    is_commutative = True
    is_number = False
    is_Rational = False

    def __new__(cls, target=-1, superscript=-1, **kwargs):
        return sp.Expr.__new__(cls, sp.sympify(target), sp.sympify(superscript), **kwargs)

    @property
    def target(self):
        return self.args[0]

    @property
    def superscript(self):
        return self.args[1]

    def _latex(self, printer, *args):
        result = "{\partial"
        if self.superscript >= 0:
            result += "^{(%s)}" % (self.superscript,)
        if self.target != -1:
            result += "_{%s}" % (self.target,)
        result += "}"
        return result

    @staticmethod
    def apply(expr, argument):
        """
        Returns a new expression where each 'DiffOperator' is replaced by a 'Diff' node.
        Multiplications of 'DiffOperator's are interpreted as nested application of differentiation:
        i.e. DiffOperator('x')*DiffOperator('x') is a second derivative replaced by Diff(Diff(arg, x), t)
        """
        def handleMul(mul):
            args = normalizeProduct(mul)
            diffs = [a for a in args if isinstance(a, DiffOperator)]
            if len(diffs) == 0:
                return mul
            rest = [a for a in args if not isinstance(a, DiffOperator)]
            diffs.sort(key=defaultDiffSortKey)
            result = argument
            for d in reversed(diffs):
                result = Diff(result, target=d.target, superscript=d.superscript)
            return prod(rest) * result

        expr = expr.expand()
        if expr.func == sp.Mul or expr.func == sp.Pow:
            return handleMul(expr)
        elif expr.func == sp.Add:
            return expr.func(*[handleMul(a) for a in expr.args])
        else:
            return expr

# ----------------------------------------------------------------------------------------------------------------------


def derivativeTerms(expr):
    """
    Returns set of all derivatives in an expression
    this is different from `expr.atoms(Diff)` when nested derivatives are in the expression,
    since this function only returns the outer derivatives
    """
    result = set()

    def visit(e):
        if isinstance(e, Diff):
            result.add(e)
        else:
            for a in e.args:
                visit(a)
    visit(expr)
    return result


def collectDerivatives(expr):
    """Rewrites expression into a sum of distinct derivatives with prefactors"""
    return expr.collect(derivativeTerms(expr))


def createNestedDiff(*args, arg=None):
    """Shortcut to create nested derivatives"""
    assert arg is not None
    args = sorted(args, reverse=True)
    res = arg
    for i in args:
        res = Diff(res, i)
    return res


def expandUsingLinearity(expr, functions=None, constants=None):
    """
    Expands all derivative nodes by applying Diff.splitLinear
    :param expr: expression containing derivatives
    :param functions: sequence of symbols that are considered functions and can not be pulled before the derivative.
                      if None, all symbols are viewed as functions
    :param constants: sequence of symbols which are considered constants and can be pulled before the derivative
    """
    if functions is None:
        functions = expr.atoms(sp.Symbol)
        if constants is not None:
            functions.difference_update(constants)

    if isinstance(expr, Diff):
        arg = expandUsingLinearity(expr.arg, functions)
        if hasattr(arg, 'func') and arg.func == sp.Add:
            result = 0
            for a in arg.args:
                result += Diff(a, target=expr.target, superscript=expr.superscript).splitLinear(functions)
            return result
        else:
            diff = Diff(arg, target=expr.target, superscript=expr.superscript)
            if diff == 0:
                return 0
            else:
                return diff.splitLinear(functions)
    else:
        newArgs = [expandUsingLinearity(e, functions) for e in expr.args]
        result = sp.expand(expr.func(*newArgs) if newArgs else expr)
        return result


def fullDiffExpand(expr, functions=None, constants=None):
    if functions is None:
        functions = expr.atoms(sp.Symbol)
        if constants is not None:
            functions.difference_update(constants)

    def visit(e):
        e = e.expand()

        if e.func == Diff:
            result = 0
            diffArgs = {'target': e.target, 'superscript': e.superscript}
            diffInner = e.args[0]
            diffInner = visit(diffInner)
            for term in diffInner.args if diffInner.func == sp.Add else [diffInner]:
                independentTerms = 1
                dependentTerms = []
                for factor in normalizeProduct(term):
                    if factor in functions or isinstance(factor, Diff):
                        dependentTerms.append(factor)
                    else:
                        independentTerms *= factor
                for i in range(len(dependentTerms)):
                    dependentTerm = dependentTerms[i]
                    otherDependentTerms = dependentTerms[:i] + dependentTerms[i+1:]
                    processedDiff = normalizeDiffOrder(Diff(dependentTerm, **diffArgs))
                    result += independentTerms * prod(otherDependentTerms) * processedDiff
            return result
        else:
            newArgs = [visit(arg) for arg in e.args]
            return e.func(*newArgs) if newArgs else e

    if isinstance(expr, sp.Matrix):
        return expr.applyfunc(visit)
    else:
        return visit(expr)


def normalizeDiffOrder(expression, functions=None, constants=None, sortKey=defaultDiffSortKey):
    """Assumes order of differentiation can be exchanged. Changes the order of nested Diffs to a standard order defined
    by the sorting key 'sortKey' such that the derivative terms can be further simplified """
    def visit(expr):
        if isinstance(expr, Diff):
            nodes = [expr]
            while isinstance(nodes[-1].arg, Diff):
                nodes.append(nodes[-1].arg)

            processedArg = visit(nodes[-1].arg)
            nodes.sort(key=sortKey)

            result = processedArg
            for d in reversed(nodes):
                result = Diff(result, target=d.target, superscript=d.superscript)
            return result
        else:
            newArgs = [visit(e) for e in expr.args]
            return expr.func(*newArgs) if newArgs else expr

    expression = expandUsingLinearity(expression.expand(), functions, constants).expand()
    return visit(expression)


def expandUsingProductRule(expr):
    """Fully expands all derivatives by applying product rule"""
    if isinstance(expr, Diff):
        arg = expandUsingProductRule(expr.args[0])
        if arg.func == sp.Add:
            newArgs = [Diff(e, target=expr.target, superscript=expr.superscript)
                       for e in arg.args]
            return sp.Add(*newArgs)
        if arg.func not in (sp.Mul, sp.Pow):
            return Diff(arg, target=expr.target, superscript=expr.superscript)
        else:
            prodList = normalizeProduct(arg)
            result = 0
            for i in range(len(prodList)):
                preFactor = prod(prodList[j] for j in range(len(prodList)) if i != j)
                result += preFactor * Diff(prodList[i], target=expr.target, superscript=expr.superscript)
            return result
    else:
        newArgs = [expandUsingProductRule(e) for e in expr.args]
        return expr.func(*newArgs) if newArgs else expr


def combineUsingProductRule(expr):
    """Inverse product rule"""

    def exprToDiffDecomposition(expr):
        """Decomposes a sp.Add node containing CeDiffs into:
        diffDict: maps (target, superscript) -> [ (preFactor, argument), ... ]
        i.e.  a partial(b) ( a is prefactor, b is argument)
            in case of partial(a) partial(b) two entries are created  (0.5 partial(a), b), (0.5 partial(b), a)
        """
        DiffInfo = namedtuple("DiffInfo", ["target", "superscript"])

        class DiffSplit:
            def __init__(self, preFactor, argument):
                self.preFactor = preFactor
                self.argument = argument

            def __repr__(self):
                return str((self.preFactor, self.argument))

        assert isinstance(expr, sp.Add)
        diffDict = defaultdict(list)
        rest = 0
        for term in expr.args:
            if isinstance(term, Diff):
                diffDict[DiffInfo(term.target, term.superscript)].append(DiffSplit(1, term.arg))
            else:
                mulArgs = normalizeProduct(term)
                diffs = [d for d in mulArgs if isinstance(d, Diff)]
                factor = prod(d for d in mulArgs if not isinstance(d, Diff))
                if len(diffs) == 0:
                    rest += factor
                else:
                    for i, diff in enumerate(diffs):
                        allButCurrent = [d for j, d in enumerate(diffs) if i != j]
                        preFactor = factor * prod(allButCurrent) * sp.Rational(1, len(diffs))
                        diffDict[DiffInfo(diff.target, diff.superscript)].append(DiffSplit(preFactor, diff.arg))

        return diffDict, rest

    def matchDiffSplits(own, other):
        ownFac = own.preFactor / other.argument
        otherFac = other.preFactor / own.argument

        if sp.count_ops(ownFac) > sp.count_ops(own.preFactor) or sp.count_ops(otherFac) > sp.count_ops(other.preFactor):
            return None

        newOtherFactor = ownFac - otherFac
        return newOtherFactor

    def processDiffList(diffList, label, superscript):
        if len(diffList) == 0:
            return 0
        elif len(diffList) == 1:
            return diffList[0].preFactor * Diff(diffList[0].argument, label, superscript)

        result = 0
        matches = []
        for i in range(1, len(diffList)):
            matchResult = matchDiffSplits(diffList[i], diffList[0])
            if matchResult is not None:
                matches.append((i, matchResult))

        if len(matches) == 0:
            result += diffList[0].preFactor * Diff(diffList[0].argument, label, superscript)
        else:
            otherIdx, matchResult = sorted(matches, key=lambda e: sp.count_ops(e[1]))[0]
            newArgument = diffList[0].argument * diffList[otherIdx].argument
            result += (diffList[0].preFactor / diffList[otherIdx].argument) * Diff(newArgument, label, superscript)
            if matchResult == 0:
                del diffList[otherIdx]
            else:
                diffList[otherIdx].preFactor = matchResult * diffList[0].argument
        result += processDiffList(diffList[1:], label, superscript)
        return result

    expr = expr.expand()
    if isinstance(expr, sp.Add):
        diffDict, rest = exprToDiffDecomposition(expr)
        for (label, superscript), diffList in diffDict.items():
            rest += processDiffList(diffList, label, superscript)
        return rest
    else:
        newArgs = [combineUsingProductRule(e) for e in expr.args]
        return expr.func(*newArgs) if newArgs else expr


def replaceDiff(expr, replacementDict):
    """replacementDict: maps variable (target) to a new Differential operator"""

    def visit(e):
        if isinstance(e, Diff):
            if e.target in replacementDict:
                return DiffOperator.apply(replacementDict[e.target], visit(e.arg))
        newArgs = [visit(arg) for arg in e.args]
        return e.func(*newArgs) if newArgs else e

    return visit(expr)


def zeroDiffs(expr, label):
    """Replaces all differentials with the given target by 0"""
    def visit(e):
        if isinstance(e, Diff):
            if e.target == label:
                return 0
        newArgs = [visit(arg) for arg in e.args]
        return e.func(*newArgs) if newArgs else e
    return visit(expr)


def evaluateDiffs(expr, var=None):
    """Replaces Diff nodes by sp.diff , the free variable is either the target (if var=None) otherwise
    the specified var"""
    if isinstance(expr, Diff):
        if var is None:
            var = expr.target
        return sp.diff(evaluateDiffs(expr.arg, var), var)
    else:
        newArgs = [evaluateDiffs(arg, var) for arg in expr.args]
        return expr.func(*newArgs) if newArgs else expr


def functionalDerivative(functional, v, constants=None):
    """
    Computes functional derivative of functional with respect to v using Euler-Lagrange equation

    .. math ::

        \frac{\delta F}{\delta v} =
                \frac{\partial F}{\partial v} - \nabla \cdot \frac{\partial F}{\partial \nabla v}

    - assumes that gradients are represented by Diff() node (from Chapman Enskog module)
    - Diff(Diff(r)) represents the divergence of r
    - the constants parameter is a list with symbols not affected by the derivative. This is used for simplification
      of the derivative terms.
    """
    diffs = functional.atoms(Diff)
    nonDiffPart = functional.subs({d: sp.Dummy() for d in diffs})

    partialF_partialV = sp.diff(nonDiffPart, v)

    gradientPart = 0
    for diffObj in diffs:
        if diffObj.args[0] != v:
            continue
        dummy = sp.Dummy()
        partialF_partialGradV = functional.subs(diffObj, dummy).diff(dummy).subs(dummy, diffObj)
        gradientPart += Diff(partialF_partialGradV, target=diffObj.target, superscript=diffObj.superscript)

    result = partialF_partialV - gradientPart
    return expandUsingLinearity(result, constants=constants)
