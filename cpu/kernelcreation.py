import sympy as sp
from pystencils.transformations import resolveFieldAccesses, makeLoopOverDomain, typingFromSympyInspection, \
    typeAllEquations, getOptimalLoopOrdering, parseBasePointerInfo, moveConstantsBeforeLoop, splitInnerLoop
from pystencils.typedsymbol import TypedSymbol
from pystencils.field import Field
import pystencils.ast as ast


def createKernel(listOfEquations, functionName="kernel", typeForSymbol=None, splitGroups=()):
    """
    Creates an abstract syntax tree for a kernel function, by taking a list of update rules.

    Loops are created according to the field accesses in the equations.

    :param listOfEquations: list of sympy equations, containing accesses to :class:`pystencils.field.Field`.
           Defining the update rules of the kernel
    :param functionName: name of the generated function - only important if generated code is written out
    :param typeForSymbol: a map from symbol name to a C type specifier. If not specified all symbols are assumed to
           be of type 'double' except symbols which occur on the left hand side of equations where the
           right hand side is a sympy Boolean which are assumed to be 'bool' .
    :param splitGroups: Specification on how to split up inner loop into multiple loops. For details see
           transformation :func:`pystencils.transformation.splitInnerLoop`
    :return: :class:`pystencils.ast.KernelFunction` node
    """
    if not typeForSymbol:
        typeForSymbol = typingFromSympyInspection(listOfEquations, "double")

    def typeSymbol(term):
        if isinstance(term, Field.Access) or isinstance(term, TypedSymbol):
            return term
        elif isinstance(term, sp.Symbol):
            return TypedSymbol(term.name, typeForSymbol[term.name])
        else:
            raise ValueError("Term has to be field access or symbol")

    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    allFields = fieldsRead.union(fieldsWritten)

    for field in allFields:
        field.setReadOnly(False)
    for field in fieldsRead - fieldsWritten:
        field.setReadOnly()

    body = ast.Block(assignments)
    code = makeLoopOverDomain(body, functionName)

    if splitGroups:
        typedSplitGroups = [[typeSymbol(s) for s in splitGroup] for splitGroup in splitGroups]
        splitInnerLoop(code, typedSplitGroups)

    loopOrder = getOptimalLoopOrdering(allFields)

    basePointerInfo = [['spatialInner0'], ['spatialInner1']]
    basePointerInfos = {field.name: parseBasePointerInfo(basePointerInfo, loopOrder, field) for field in allFields}

    resolveFieldAccesses(code, fieldToBasePointerInfo=basePointerInfos)
    moveConstantsBeforeLoop(code)

    return code


def addOpenMP(astNode, schedule="static"):
    """
    Parallelizes the outer loop with OpenMP

    :param astNode: abstract syntax tree created e.g. by :func:`createKernel`
    :param schedule: OpenMP scheduling policy e.g. 'static' or 'dynamic'
    """
    assert type(astNode) is ast.KernelFunction
    body = astNode.body
    wrapperBlock = ast.PragmaBlock('#pragma omp parallel', body.takeChildNodes())
    body.append(wrapperBlock)

    outerLoops = [l for l in body.atoms(ast.LoopOverCoordinate) if l.isOutermostLoop]
    assert outerLoops, "No outer loop found"
    assert len(outerLoops) <= 1, "More than one outer loop found. Which one should be parallelized?"
    outerLoops[0].prefixLines.append("#pragma omp for schedule(%s)" % (schedule,))
