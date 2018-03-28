import operator
from functools import reduce
from collections import defaultdict, Sequence
import itertools
import warnings
import sympy as sp

from pystencils.data_types import getTypeOfExpression, getBaseType


def prod(seq):
    """Takes a sequence and returns the product of all elements"""
    return reduce(operator.mul, seq, 1)


def allIn(a, b):
    """Tests if all elements of a container 'a' are contained in 'b'"""
    return all(element in b for element in a)


def isIntegerSequence(sequence):
    try:
        [int(i) for i in sequence]
        return True
    except TypeError:
        return False


def scalarProduct(a, b):
    return sum(a_i * b_i for a_i, b_i in zip(a, b))


def equationsToMatrix(equations, degreesOfFreedom):
    return sp.Matrix(len(equations), len(degreesOfFreedom),
                     lambda row, col: equations[row].coeff(degreesOfFreedom[col]))


def kroneckerDelta(*args):
    """Kronecker delta for variable number of arguments, 
    1 if all args are equal, otherwise 0"""
    for a in args:
        if a != args[0]:
            return 0
    return 1


def multidimensionalSummation(i, dim):
    """Multidimensional summation"""
    prodArgs = [range(dim)] * i
    return itertools.product(*prodArgs)


def normalizeProduct(product):
    """
    Expects a sympy expression that can be interpreted as a product and
    - for a Mul node returns its factors ('args')
    - for a Pow node with positive integer exponent returns a list of factors
    - for other node types [product] is returned
    """
    def handlePow(power):
        if power.exp.is_integer and power.exp.is_number and power.exp > 0:
            return [power.base] * power.exp
        else:
            return [power]

    if product.func == sp.Pow:
        return handlePow(product)
    elif product.func == sp.Mul:
        result = []
        for a in product.args:
            if a.func == sp.Pow:
                result += handlePow(a)
            else:
                result.append(a)
        return result
    else:
        return [product]


def productSymmetric(*args, withDiagonal=True):
    """Similar to itertools.product but returns only values where the index is ascending i.e. values below diagonal"""
    ranges = [range(len(a)) for a in args]
    for idx in itertools.product(*ranges):
        validIndex = True
        for t in range(1, len(idx)):
            if (withDiagonal and idx[t - 1] > idx[t]) or (not withDiagonal and idx[t - 1] >= idx[t]):
                validIndex = False
                break
        if validIndex:
            yield tuple(a[i] for a, i in zip(args, idx))


def fastSubs(term, subsDict, skip=None):
    """Similar to sympy subs function.
    This version is much faster for big substitution dictionaries than sympy version"""
    def visit(expr):
        if skip and skip(expr):
            return expr
        if hasattr(expr, "fastSubs"):
            return expr.fastSubs(subsDict)
        if expr in subsDict:
            return subsDict[expr]
        if not hasattr(expr, 'args'):
            return expr
        paramList = [visit(a) for a in expr.args]
        return expr if not paramList else expr.func(*paramList)

    if len(subsDict) == 0:
        return term
    else:
        return visit(term)


def fastSubsWithNormalize(term, subsDict, normalizeFunc):
    def visit(expr):
        if expr in subsDict:
            return subsDict[expr], True
        if not hasattr(expr, 'args'):
            return expr, False

        paramList = []
        substituted = False
        for a in expr.args:
            replacedExpr, s = visit(a)
            paramList.append(replacedExpr)
            if s:
                substituted = True

        if not paramList:
            return expr, False
        else:
            if substituted:
                result, _ = visit(normalizeFunc(expr.func(*paramList)))
                return result, True
            else:
                return expr.func(*paramList), False

    if len(subsDict) == 0:
        return term
    else:
        res, _ = visit(term)
        return res


def replaceAdditive(expr, replacement, subExpression, requiredMatchReplacement=0.5, requiredMatchOriginal=None):
    """
    Transformation for replacing a given subexpression inside a sum

    Example 1:
        expr = 3*x + 3 * y
        replacement = k
        subExpression = x+y
        return = 3*k

    Example 2:
        expr = 3*x + 3 * y + z
        replacement = k
        subExpression = x+y+z
        return:
            if minimalMatchingTerms >=3 the expression would not be altered
            if smaller than 3 the result is 3*k - 2*z

    :param expr: input expression
    :param replacement: expression that is inserted for subExpression (if found)
    :param subExpression: expression to replace
    :param requiredMatchReplacement:
        - if float: the percentage of terms of the subExpression that has to be matched in order to replace
        - if integer: the total number of terms that has to be matched in order to replace
        - None: is equal to integer 1
        - if both match parameters are given, both restrictions have to be fulfilled (i.e. logical AND)
    :param requiredMatchOriginal:
        - if float: the percentage of terms of the original addition expression that has to be matched
        - if integer: the total number of terms that has to be matched in order to replace
        - None: is equal to integer 1
    :return: new expression with replacement
    """
    def normalizeMatchParameter(matchParameter, expressingLength):
        if matchParameter is None:
            return 1
        elif isinstance(matchParameter, float):
            assert 0 <= matchParameter <= 1
            res = int(matchParameter * expressingLength)
            return max(res, 1)
        elif isinstance(matchParameter, int):
            assert matchParameter > 0
            return matchParameter
        raise ValueError("Invalid parameter")

    normalizedReplacementMatch = normalizeMatchParameter(requiredMatchReplacement, len(subExpression.args))

    def visit(currentExpr):
        if currentExpr.is_Add:
            exprMaxLength = max(len(currentExpr.args), len(subExpression.args))
            normalizedCurrentExprMatch = normalizeMatchParameter(requiredMatchOriginal, exprMaxLength)
            exprCoeffs = currentExpr.as_coefficients_dict()
            subexprCoeffDict = subExpression.as_coefficients_dict()
            intersection = set(subexprCoeffDict.keys()).intersection(set(exprCoeffs))
            if len(intersection) >= max(normalizedReplacementMatch, normalizedCurrentExprMatch):
                # find common factor
                factors = defaultdict(lambda: 0)
                skips = 0
                for commonSymbol in subexprCoeffDict.keys():
                    if commonSymbol not in exprCoeffs:
                        skips += 1
                        continue
                    factor = exprCoeffs[commonSymbol] / subexprCoeffDict[commonSymbol]
                    factors[sp.simplify(factor)] += 1

                commonFactor = max(factors.items(), key=operator.itemgetter(1))[0]
                if factors[commonFactor] >= max(normalizedCurrentExprMatch, normalizedReplacementMatch):
                    return currentExpr - commonFactor * subExpression + commonFactor * replacement

        # if no subexpression was found
        paramList = [visit(a) for a in currentExpr.args]
        if not paramList:
            return currentExpr
        else:
            return currentExpr.func(*paramList, evaluate=False)

    return visit(expr)


def replaceSecondOrderProducts(expr, searchSymbols, positive=None, replaceMixed=None):
    """
    Replaces second order mixed terms like x*y by 2* ( (x+y)**2 - x**2 - y**2 )
    This makes the term longer - simplify usually is undoing these - however this
    transformation can be done to find more common sub-expressions
    :param expr: input expression
    :param searchSymbols: list of symbols that are searched for
                            Example: given [ x,y,z] terms like x*y, x*z, z*y are replaced
    :param positive: there are two ways to do this substitution, either with term
                    (x+y)**2 or (x-y)**2 . if positive=True the first version is done,
                    if positive=False the second version is done, if positive=None the
                    sign is determined by the sign of the mixed term that is replaced
    :param replaceMixed: if a list is passed here the expr x+y or x-y is replaced by a special new symbol
                         the replacement equation is added to the list
    :return:
    """
    if replaceMixed is not None:
        mixedSymbolsReplaced = set([e.lhs for e in replaceMixed])

    if expr.is_Mul:
        distinctVelTerms = set()
        nrOfVelTerms = 0
        otherFactors = 1
        for t in expr.args:
            if t in searchSymbols:
                nrOfVelTerms += 1
                distinctVelTerms.add(t)
            else:
                otherFactors *= t
        if len(distinctVelTerms) == 2 and nrOfVelTerms == 2:
            u, v = sorted(list(distinctVelTerms), key=lambda symbol: symbol.name)
            if positive is None:
                otherFactorsWithoutSymbols = otherFactors
                for s in otherFactors.atoms(sp.Symbol):
                    otherFactorsWithoutSymbols = otherFactorsWithoutSymbols.subs(s, 1)
                positive = otherFactorsWithoutSymbols.is_positive
                assert positive is not None
            sign = 1 if positive else -1
            if replaceMixed is not None:
                newSymbolStr = 'P' if positive else 'M'
                mixedSymbolName = u.name + newSymbolStr + v.name
                mixedSymbol = sp.Symbol(mixedSymbolName.replace("_", ""))
                if mixedSymbol not in mixedSymbolsReplaced:
                    mixedSymbolsReplaced.add(mixedSymbol)
                    replaceMixed.append(sp.Eq(mixedSymbol, u + sign * v))
            else:
                mixedSymbol = u + sign * v
            return sp.Rational(1, 2) * sign * otherFactors * (mixedSymbol ** 2 - u ** 2 - v ** 2)

    paramList = [replaceSecondOrderProducts(a, searchSymbols, positive, replaceMixed) for a in expr.args]
    result = expr.func(*paramList, evaluate=False) if paramList else expr
    return result


def removeHigherOrderTerms(term, order=3, symbols=None):
    """
    Removes all terms that that contain more than 'order' factors of given 'symbols'

    Example:
        >>> x, y = sp.symbols("x y")
        >>> term = x**2 * y + y**2 * x + y**3 + x + y ** 2
        >>> removeHigherOrderTerms(term, order=2, symbols=[x, y])
        x + y**2
    """
    from sympy.core.power import Pow
    from sympy.core.add import Add, Mul

    result = 0
    term = term.expand()

    if not symbols:
        symbols = sp.symbols(" ".join(["u_%d" % (i,) for i in range(3)]))
        symbols += sp.symbols(" ".join(["u_%d" % (i,) for i in range(3)]), real=True)

    def velocityFactorsInProduct(product):
        uFactorCount = 0
        if type(product) is Mul:
            for factor in product.args:
                if type(factor) == Pow:
                    if factor.args[0] in symbols:
                        uFactorCount += factor.args[1]
                if factor in symbols:
                    uFactorCount += 1
        elif type(product) is Pow:
            if product.args[0] in symbols:
                uFactorCount += product.args[1]
        return uFactorCount

    if type(term) == Mul or type(term) == Pow:
        if velocityFactorsInProduct(term) <= order:
            return term
        else:
            return sp.Rational(0, 1)

    if type(term) != Add:
        return term

    for sumTerm in term.args:
        if velocityFactorsInProduct(sumTerm) <= order:
            result += sumTerm
    return result


def completeTheSquare(expr, symbolToComplete, newVariable):
    """
    Transforms second order polynomial into only squared part i.e.
        a*symbolToComplete**2 + b*symbolToComplete + c
          is transformed into
        newVariable**2 + d

    returns replacedExpr, "a tuple to to replace newVariable such that old expr comes out again"

    if given expr is not a second order polynomial:
        return expr, None
    """
    p = sp.Poly(expr, symbolToComplete)
    coeffs = p.all_coeffs()
    if len(coeffs) != 3:
        return expr, None
    a, b, _ = coeffs
    expr = expr.subs(symbolToComplete, newVariable - b / (2 * a))
    return sp.simplify(expr), (newVariable, symbolToComplete + b / (2 * a))


def makeExponentialFuncArgumentSquares(expr, variablesToCompleteSquares):
    """Completes squares in arguments of exponential which makes them simpler to integrate
    Very useful for integrating Maxwell-Boltzmann and its moment generating function"""
    expr = sp.simplify(expr)
    dim = len(variablesToCompleteSquares)
    dummies = [sp.Dummy() for i in range(dim)]

    def visit(term):
        if term.func == sp.exp:
            expArg = term.args[0]
            for i in range(dim):
                expArg, substitution = completeTheSquare(expArg, variablesToCompleteSquares[i], dummies[i])
            return sp.exp(sp.expand(expArg))
        else:
            paramList = [visit(a) for a in term.args]
            if not paramList:
                return term
            else:
                return term.func(*paramList)

    result = visit(expr)
    for i in range(dim):
        result = result.subs(dummies[i], variablesToCompleteSquares[i])
    return result


def pow2mul(expr):
    """
    Convert integer powers in an expression to Muls, like a**2 => a*a.
    """
    pows = list(expr.atoms(sp.Pow))
    if any(not e.is_Integer for b, e in (i.as_base_exp() for i in pows)):
        raise ValueError("A power contains a non-integer exponent")
    repl = zip(pows, (sp.Mul(*[b]*e, evaluate=False) for b, e in (i.as_base_exp() for i in pows)))
    return expr.subs(repl)


def extractMostCommonFactor(term):
    """Processes a sum of fractions: determines the most common factor and splits term in common factor and rest"""
    import operator
    from collections import Counter
    from sympy.functions import Abs

    coeffDict = term.as_coefficients_dict()
    counter = Counter([Abs(v) for v in coeffDict.values()])
    commonFactor, occurrences = max(counter.items(), key=operator.itemgetter(1))
    if occurrences == 1 and (1 in counter):
        commonFactor = 1
    return commonFactor, term / commonFactor


def mostCommonTermFactorization(term):
    commonFactor, term = extractMostCommonFactor(term)

    factorization = sp.factor(term)
    if factorization.is_Mul:
        symbolsInFactorization = []
        constantsInFactorization = 1
        for arg in factorization.args:
            if len(arg.atoms(sp.Symbol)) == 0:
                constantsInFactorization *= arg
            else:
                symbolsInFactorization.append(arg)
        if len(symbolsInFactorization) <= 1:
            return sp.Mul(commonFactor, term, evaluate=False)
        else:
            args = symbolsInFactorization[:-1] + [constantsInFactorization * symbolsInFactorization[-1]]
            return sp.Mul(commonFactor, *args)
    else:
        return sp.Mul(commonFactor, term, evaluate=False)


def countNumberOfOperations(term, onlyType='real'):
    """
    Counts the number of additions, multiplications and division
    :param term: a sympy term, equation or sequence of terms/equations
    :param onlyType: 'real' or 'int' to count only operations on these types, or None for all
    :return: a dictionary with 'adds', 'muls' and 'divs' keys
    """
    result = {'adds': 0, 'muls': 0, 'divs': 0}

    if isinstance(term, Sequence):
        for element in term:
            r = countNumberOfOperations(element, onlyType)
            for operationName in result.keys():
                result[operationName] += r[operationName]
        return result
    elif isinstance(term, sp.Eq):
        term = term.rhs

    term = term.evalf()

    def checkType(e):
        if onlyType is None:
            return True
        try:
            type = getBaseType(getTypeOfExpression(e))
        except ValueError:
            return False
        if onlyType == 'int' and (type.is_int() or type.is_uint()):
            return True
        if onlyType == 'real' and (type.is_float()):
            return True
        else:
            return type == onlyType

    def visit(t):
        visitChildren = True
        if t.func is sp.Add:
            if checkType(t):
                result['adds'] += len(t.args) - 1
        elif t.func is sp.Mul:
            if checkType(t):
                result['muls'] += len(t.args) - 1
                for a in t.args:
                    if a == 1 or a == -1:
                        result['muls'] -= 1
        elif t.func is sp.Float:
            pass
        elif isinstance(t, sp.Symbol):
            visitChildren = False
        elif isinstance(t, sp.tensor.Indexed):
            visitChildren = False
        elif t.is_integer:
            pass
        elif t.func is sp.Pow:
            if checkType(t.args[0]):
                visitChildren = False
                if t.exp.is_integer and t.exp.is_number:
                    if t.exp >= 0:
                        result['muls'] += int(t.exp) - 1
                    else:
                        result['muls'] -= 1
                        result['divs'] += 1
                        result['muls'] += (-int(t.exp)) - 1
            else:
                warnings.warn("Counting operations: only integer exponents are supported in Pow, "
                              "counting will be inaccurate")
        else:
            warnings.warn("Unknown sympy node of type " + str(t.func) + " counting will be inaccurate")

        if visitChildren:
            for a in t.args:
                visit(a)

    visit(term)
    return result


def countNumberOfOperationsInAst(ast):
    """Counts number of operations in an abstract syntax tree, see also :func:`countNumberOfOperations`"""
    from pystencils.astnodes import SympyAssignment
    result = {'adds': 0, 'muls': 0, 'divs': 0}

    def visit(node):
        if isinstance(node, SympyAssignment):
            r = countNumberOfOperations(node.rhs)
            result['adds'] += r['adds']
            result['muls'] += r['muls']
            result['divs'] += r['divs']
        else:
            for arg in node.args:
                visit(arg)
    visit(ast)
    return result


def matrixFromColumnVectors(columnVectors):
    """Creates a sympy matrix from column vectors.
        :param columnVectors: nested sequence - i.e. a sequence of column vectors
    """
    c = columnVectors
    return sp.Matrix([list(c[i]) for i in range(len(c))]).transpose()


def commonDenominator(expr):
    denominators = [r.q for r in expr.atoms(sp.Rational)]
    return sp.lcm(denominators)


def getSymmetricPart(term, vars):
    """
    Returns the symmetric part of a sympy expressions.

    :param term: sympy expression, labeled here as :math:`f`
    :param vars: sequence of symbols which are considered as degrees of freedom, labeled here as :math:`x_0, x_1,...`
    :returns: :math:`\frac{1}{2} [ f(x_0, x_1, ..) + f(-x_0, -x_1) ]`
    """
    substitutionDict = {e: -e for e in vars}
    return sp.Rational(1, 2) * (term + term.subs(substitutionDict))


def sortEquationsTopologically(equationSequence):
    res = sp.cse_main.reps_toposort([[e.lhs, e.rhs] for e in equationSequence])
    return [sp.Eq(a, b) for a, b in res]


def getEquationsFromFunction(func, **kwargs):
    """
    Mechanism to simplify the generation of a list of sympy equations. 
    Introduces a special "assignment operator" written as "@=". Each line containing this operator gives an
    equation in the result list. Note that executing this function normally yields an error.
    
    Additionally the shortcut object 'S' is available to quickly create new sympy symbols.
    
    Example:
        
    >>> def myKernel():
    ...     from pystencils import Field
    ...     f = Field.createGeneric('f', spatialDimensions=2, indexDimensions=0)
    ...     g = f.newFieldWithDifferentName('g')
    ...     
    ...     S.neighbors @= f[0,1] + f[1,0]
    ...     g[0,0]      @= S.neighbors + f[0,0]
    >>> getEquationsFromFunction(myKernel)
    [Eq(neighbors, f_E + f_N), Eq(g_C, f_C + neighbors)]
    """
    import inspect
    import re

    class SymbolCreator:
        def __getattribute__(self, name):
            return sp.Symbol(name)

    assignmentRegexp = re.compile(r'(\s*)(.+?)@=(.*)')
    whitespaceRegexp = re.compile(r'(\s*)(.*)')
    sourceLines = inspect.getsourcelines(func)[0]

    # determine indentation
    firstCodeLine = sourceLines[1]
    matchRes = whitespaceRegexp.match(firstCodeLine)
    assert matchRes, "First line is not indented"
    numWhitespaces = len(matchRes.group(1))

    for i in range(1, len(sourceLines)):
        sourceLine = sourceLines[i][numWhitespaces:]
        if 'return' in sourceLine:
            raise ValueError("Function may not have a return statement!")
        matchRes = assignmentRegexp.match(sourceLine)
        if matchRes:
            sourceLine = "%s_result.append(Eq(%s, %s))\n" % matchRes.groups()
        sourceLines[i] = sourceLine

    code = "".join(sourceLines[1:])
    result = []
    localsDict = {'_result': result,
                  'Eq': sp.Eq,
                  'S': SymbolCreator()}
    localsDict.update(kwargs)
    globalsDict = inspect.stack()[1][0].f_globals.copy()
    globalsDict.update(inspect.stack()[1][0].f_locals)

    exec(code, globalsDict, localsDict)
    return result
