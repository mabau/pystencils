import sympy as sp
import warnings

from pystencils.sympyextensions import fast_subs
from pystencils.transformations import filteredTreeIteration
from pystencils.data_types import TypedSymbol, VectorType, BasicType, getTypeOfExpression, castFunc, collateTypes, \
    PointerType
import pystencils.astnodes as ast


def vectorize(astNode, vectorWidth=4):
    vectorizeInnerLoopsAndAdaptLoadStores(astNode, vectorWidth)
    insertVectorCasts(astNode)


def vectorizeInnerLoopsAndAdaptLoadStores(astNode, vectorWidth=4):
    """
    Goes over all innermost loops, changes increment to vector width and replaces field accesses by vector type if
        - loop bounds are constant
        - loop range is a multiple of vector width
    """
    innerLoops = [n for n in astNode.atoms(ast.LoopOverCoordinate) if n.isInnermostLoop]

    for loopNode in innerLoops:
        loopRange = loopNode.stop - loopNode.start

        # Check restrictions
        if isinstance(loopRange, sp.Basic) and not loopRange.is_integer:
            warnings.warn("Currently only loops with fixed ranges can be vectorized - skipping loop")
            continue
        if loopRange % vectorWidth != 0 or loopNode.step != 1:
            warnings.warn("Currently only loops with loop bounds that are multiples "
                          "of vectorization width can be vectorized")
            continue

        # Find all array accesses (indexed) that depend on the loop counter as offset
        loopCounterSymbol = ast.LoopOverCoordinate.getLoopCounterSymbol(loopNode.coordinateToLoopOver)
        substitutions = {}
        successful = True
        for indexed in loopNode.atoms(sp.Indexed):
            base, index = indexed.args
            if loopCounterSymbol in index.atoms(sp.Symbol):
                loopCounterIsOffset = loopCounterSymbol not in (index - loopCounterSymbol).atoms()
                if not loopCounterIsOffset:
                    successful = False
                    break
                typedSymbol = base.label
                assert type(typedSymbol.dtype) is PointerType, "Type of access is " + str(typedSymbol.dtype) + ", " + str(indexed)
                substitutions[indexed] = castFunc(indexed, VectorType(typedSymbol.dtype.baseType, vectorWidth))
        if not successful:
            warnings.warn("Could not vectorize loop because of non-consecutive memory access")
            continue

        loopNode.step = vectorWidth
        loopNode.subs(substitutions)


def insertVectorCasts(astNode):
    """
    Inserts necessary casts from scalar values to vector values
    """
    def visitExpr(expr):
        if expr.func in (sp.Add, sp.Mul) or (isinstance(expr, sp.Rel) and not expr.func == castFunc) or \
                isinstance(expr, sp.boolalg.BooleanFunction):
            newArgs = [visitExpr(a) for a in expr.args]
            argTypes = [getTypeOfExpression(a) for a in newArgs]
            if not any(type(t) is VectorType for t in argTypes):
                return expr
            else:
                targetType = collateTypes(argTypes)
                castedArgs = [castFunc(a, targetType) if t != targetType else a
                              for a, t in zip(newArgs, argTypes)]
                return expr.func(*castedArgs)
        elif expr.func is sp.Pow:
            newArg = visitExpr(expr.args[0])
            return sp.Pow(newArg, expr.args[1])
        elif expr.func == sp.Piecewise:
            newResults = [visitExpr(a[0]) for a in expr.args]
            newConditions = [visitExpr(a[1]) for a in expr.args]
            typesOfResults = [getTypeOfExpression(a) for a in newResults]
            typesOfConditions = [getTypeOfExpression(a) for a in newConditions]

            resultTargetType = getTypeOfExpression(expr)
            conditionTargetType = collateTypes(typesOfConditions)
            if type(conditionTargetType) is VectorType and type(resultTargetType) is not VectorType:
                resultTargetType = VectorType(resultTargetType, width=conditionTargetType.width)

            castedResults = [castFunc(a, resultTargetType) if t != resultTargetType else a
                             for a, t in zip(newResults, typesOfResults)]

            castedConditions = [castFunc(a, conditionTargetType) if t != conditionTargetType and a != True else a
                                for a, t in zip(newConditions, typesOfConditions)]

            return sp.Piecewise(*[(r, c) for r, c in zip(castedResults, castedConditions)])
        else:
            return expr

    substitutionDict = {}
    for asmt in filteredTreeIteration(astNode, ast.SympyAssignment):
        subsExpr = fast_subs(asmt.rhs, substitutionDict, skip=lambda e: isinstance(e, ast.ResolvedFieldAccess))
        asmt.rhs = visitExpr(subsExpr)
        rhsType = getTypeOfExpression(asmt.rhs)
        if isinstance(asmt.lhs, TypedSymbol):
            lhsType = asmt.lhs.dtype
            if type(rhsType) is VectorType and type(lhsType) is not VectorType:
                newLhsType = VectorType(lhsType, rhsType.width)
                newLhs = TypedSymbol(asmt.lhs.name, newLhsType)
                substitutionDict[asmt.lhs] = newLhs
                asmt.lhs = newLhs
        elif asmt.lhs.func == castFunc:
            lhsType = asmt.lhs.args[1]
            if type(lhsType) is VectorType and type(rhsType) is not VectorType:
                asmt.rhs = castFunc(asmt.rhs, lhsType)

