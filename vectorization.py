import sympy as sp
import warnings

from pystencils.transformations import filteredTreeIteration
from pystencils.data_types import TypedSymbol, VectorType, BasicType, getTypeOfExpression, castFunc, collateTypes
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

        loopNode.step = vectorWidth

        # All field accesses depending on loop coordinate are changed to vector type
        fieldAccesses = [n for n in loopNode.atoms(ast.ResolvedFieldAccess)]
        substitutions = {fa: castFunc(fa, VectorType(BasicType(fa.field.dtype), vectorWidth)) for fa in fieldAccesses}
        loopNode.subs(substitutions)


def insertVectorCasts(astNode):
    """
    Inserts necessary casts from scalar values to vector values
    """
    def visitExpr(expr):
        if expr.func in (sp.Add, sp.Mul) or (isinstance(expr, sp.Rel) and not expr.func == castFunc):
            newArgs = [visitExpr(a) for a in expr.args]
            argTypes = [getTypeOfExpression(a) for a in newArgs]
            if not any(type(t) is VectorType for t in argTypes):
                return expr
            else:
                targetType = collateTypes(argTypes)
                castedArgs = [castFunc(a, targetType) if t != targetType else a
                              for a, t in zip(newArgs, argTypes)]
                return expr.func(*castedArgs)
        elif expr.func == sp.Piecewise:
            newResults = [visitExpr(a[0]) for a in expr.args]
            newConditions = [visitExpr(a[1]) for a in expr.args]
            typesOfResults = [getTypeOfExpression(a) for a in newResults]
            typesOfConditions = [getTypeOfExpression(a) for a in newConditions]

            resultTargetType = getTypeOfExpression(expr)
            castedResults = [castFunc(a, resultTargetType) if t != resultTargetType else a
                             for a, t in zip(newResults, typesOfResults)]

            conditionTargetType = collateTypes(typesOfConditions)
            castedConditions = [castFunc(a, conditionTargetType) if t != conditionTargetType and a != True else a
                                for a, t in zip(newConditions, typesOfConditions)]

            return sp.Piecewise(*[(r, c) for r, c in zip(castedResults, castedConditions)])
        else:
            return expr

    substitutionDict = {}
    for asmt in filteredTreeIteration(astNode, ast.SympyAssignment):
        subsExpr = asmt.rhs.subs(substitutionDict)
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





