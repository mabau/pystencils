from pystencils.gpucuda.indexing import BlockIndexing
from pystencils.transformations import resolveFieldAccesses, typeAllEquations, parseBasePointerInfo, getCommonShape
from pystencils.astnodes import Block, KernelFunction, SympyAssignment
from pystencils.types import TypedSymbol, BasicType, StructType
from pystencils import Field


def createCUDAKernel(listOfEquations, functionName="kernel", typeForSymbol=None, indexingCreator=BlockIndexing,
                     ghostLayers=None):
    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    allFields = fieldsRead.union(fieldsWritten)
    readOnlyFields = set([f.name for f in fieldsRead - fieldsWritten])

    fieldAccesses = set()
    for eq in listOfEquations:
        fieldAccesses.update(eq.atoms(Field.Access))

    commonShape = getCommonShape(allFields)
    if ghostLayers is None:
        requiredGhostLayers = max([fa.requiredGhostLayers for fa in fieldAccesses])
        ghostLayers = [(requiredGhostLayers, requiredGhostLayers)] * len(commonShape)
    if isinstance(ghostLayers, int):
        ghostLayers = [(ghostLayers, ghostLayers)] * len(commonShape)

    indexing = indexingCreator(field=list(fieldsRead)[0], ghostLayers=ghostLayers)

    block = Block(assignments)
    block = indexing.guard(block, commonShape)
    ast = KernelFunction(block, allFields, functionName)
    ast.globalVariables.update(indexing.indexVariables)

    coordMapping = indexing.coordinates
    basePointerInfo = [['spatialInner0']]
    basePointerInfos = {f.name: parseBasePointerInfo(basePointerInfo, [2, 1, 0], f) for f in allFields}

    coordMapping = {f.name: coordMapping for f in allFields}
    resolveFieldAccesses(ast, readOnlyFields, fieldToFixedCoordinates=coordMapping,
                         fieldToBasePointerInfo=basePointerInfos)
    # add the function which determines #blocks and #threads as additional member to KernelFunction node
    # this is used by the jit
    ast.indexing = indexing
    return ast


def createdIndexedCUDAKernel(listOfEquations, indexFields, functionName="kernel", typeForSymbol=None,
                             coordinateNames=('x', 'y', 'z'), indexingCreator=BlockIndexing):
    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    allFields = fieldsRead.union(fieldsWritten)
    readOnlyFields = set([f.name for f in fieldsRead - fieldsWritten])

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

    idxField = list(indexFields)[0]
    indexing = indexingCreator(field=idxField, ghostLayers=[(0, 0)] * len(idxField.shape))

    functionBody = Block(coordinateSymbolAssignments + assignments)
    functionBody = indexing.guard(functionBody, getCommonShape(indexFields))
    ast = KernelFunction(functionBody, allFields, functionName)
    ast.globalVariables.update(indexing.indexVariables)

    coordMapping = indexing.coordinates
    basePointerInfo = [['spatialInner0']]
    basePointerInfos = {f.name: parseBasePointerInfo(basePointerInfo, [2, 1, 0], f) for f in allFields}

    coordMapping = {f.name: coordMapping for f in indexFields}
    coordMapping.update({f.name: coordinateTypedSymbols for f in nonIndexFields})
    resolveFieldAccesses(ast, readOnlyFields, fieldToFixedCoordinates=coordMapping,
                         fieldToBasePointerInfo=basePointerInfos)
    # add the function which determines #blocks and #threads as additional member to KernelFunction node
    # this is used by the jit
    ast.indexing = indexing
    return ast
