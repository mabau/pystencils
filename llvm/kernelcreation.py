import sympy as sp
from pystencils.astnodes import SympyAssignment, Block, LoopOverCoordinate, KernelFunction
from pystencils.transformations import resolveFieldAccesses, makeLoopOverDomain, typingFromSympyInspection, \
    typeAllEquations, getOptimalLoopOrdering, parseBasePointerInfo, moveConstantsBeforeLoop, splitInnerLoop, insertCasts#, \
    #desympy_ast, insert_casts
from pystencils.data_types import TypedSymbol, BasicType, StructType
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

    basePointerInfo = []
    for i in range(len(loopOrder)):
        basePointerInfo.append(['spatialInner%d' % i])
    basePointerInfos = {field.name: parseBasePointerInfo(basePointerInfo, loopOrder, field) for field in allFields}

    resolveFieldAccesses(code, readOnlyFields, fieldToBasePointerInfo=basePointerInfos)
    moveConstantsBeforeLoop(code)

    #print('Ast:')
    #print(code)
    #desympy_ast(code)
    #print('Desympied ast:')
    #print(code)
    #insert_casts(code)
    print(code)
    code = insertCasts(code)
    print(code)
    return code


def createIndexedKernel(listOfEquations, indexFields, functionName="kernel", typeForSymbol=None,
                        coordinateNames=('x', 'y', 'z')):
    """
    Similar to :func:`createKernel`, but here not all cells of a field are updated but only cells with
    coordinates which are stored in an index field. This traversal method can e.g. be used for boundary handling.

    The coordinates are stored in a separated indexField, which is a one dimensional array with struct data type.
    This struct has to contain fields named 'x', 'y' and for 3D fields ('z'). These names are configurable with the
    'coordinateNames' parameter. The struct can have also other fields that can be read and written in the kernel, for
    example boundary parameters.

    :param listOfEquations: list of update equations or AST nodes
    :param indexFields: list of index fields, i.e. 1D fields with struct data type
    :param typeForSymbol: see documentation of :func:`createKernel`
    :param functionName: see documentation of :func:`createKernel`
    :param coordinateNames: name of the coordinate fields in the struct data type
    :return: abstract syntax tree
    """
    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    allFields = fieldsRead.union(fieldsWritten)

    for indexField in indexFields:
        indexField.isIndexField = True
        assert indexField.spatialDimensions == 1, "Index fields have to be 1D"

    nonIndexFields = [f for f in allFields if f not in indexFields]
    spatialCoordinates = {f.spatialDimensions for f in nonIndexFields}
    assert len(spatialCoordinates) == 1, "Non-index fields do not have the same number of spatial coordinates"
    spatialCoordinates = list(spatialCoordinates)[0]

    def getCoordinateSymbolAssignment(name):
        for indexField in indexFields:
            assert isinstance(indexField.dtype, StructType), "Index fields have to have a struct datatype"
            dataType = indexField.dtype
            if dataType.hasElement(name):
                rhs = indexField[0](name)
                lhs = TypedSymbol(name, BasicType(dataType.getElementType(name)))
                return SympyAssignment(lhs, rhs)
        raise ValueError("Index %s not found in any of the passed index fields" % (name,))

    coordinateSymbolAssignments = [getCoordinateSymbolAssignment(n) for n in coordinateNames[:spatialCoordinates]]
    coordinateTypedSymbols = [eq.lhs for eq in coordinateSymbolAssignments]
    assignments = coordinateSymbolAssignments + assignments

    # make 1D loop over index fields
    loopBody = Block([])
    loopNode = LoopOverCoordinate(loopBody, coordinateToLoopOver=0, start=0, stop=indexFields[0].shape[0])

    for assignment in assignments:
        loopBody.append(assignment)

    functionBody = Block([loopNode])
    ast = KernelFunction(functionBody, allFields, functionName)

    fixedCoordinateMapping = {f.name: coordinateTypedSymbols for f in nonIndexFields}
    resolveFieldAccesses(ast, set(['indexField']), fieldToFixedCoordinates=fixedCoordinateMapping)
    moveConstantsBeforeLoop(ast)

    desympy_ast(ast)
    insert_casts(ast)

    return ast
