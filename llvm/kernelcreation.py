import sympy as sp
from pystencils.transformations import resolveFieldAccesses, makeLoopOverDomain, typingFromSympyInspection, \
    typeAllEquations, getOptimalLoopOrdering, parseBasePointerInfo, moveConstantsBeforeLoop, splitInnerLoop, \
    desympy_ast, insert_casts
from pystencils.types import TypedSymbol
from pystencils.field import Field
import pystencils.astnodes as ast


def createKernel(listOfEquations, functionName="kernel", typeForSymbol=None, splitGroups=(),
                 iterationSlice=None, ghostLayers=None):
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
    :param iterationSlice: if not None, iteration is done only over this slice of the field
    :param ghostLayers: a sequence of pairs for each coordinate with lower and upper nr of ghost layers
                        if None, the number of ghost layers is determined automatically and assumed to be equal for a
                        all dimensions

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

    readOnlyFields = set([f.name for f in fieldsRead - fieldsWritten])

    body = ast.Block(assignments)
    loopOrder = getOptimalLoopOrdering(allFields)
    code = makeLoopOverDomain(body, functionName, iterationSlice=iterationSlice,
                              ghostLayers=ghostLayers, loopOrder=loopOrder)

    if splitGroups:
        typedSplitGroups = [[typeSymbol(s) for s in splitGroup] for splitGroup in splitGroups]
        splitInnerLoop(code, typedSplitGroups)

    basePointerInfo = [['spatialInner0'], ['spatialInner1']]
    basePointerInfos = {field.name: parseBasePointerInfo(basePointerInfo, loopOrder, field) for field in allFields}

    resolveFieldAccesses(code, readOnlyFields, fieldToBasePointerInfo=basePointerInfos)
    moveConstantsBeforeLoop(code)

    # print(code)
    desympy_ast(code)
    # print(code)
    insert_casts(code)

    return code
