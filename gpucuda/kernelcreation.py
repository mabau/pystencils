import sympy as sp

from pystencils.transformations import resolveFieldAccesses, typeAllEquations, parseBasePointerInfo
from pystencils.astnodes import Block, KernelFunction, SympyAssignment
from pystencils import Field
from pystencils.types import TypedSymbol, BasicType, StructType

BLOCK_IDX = list(sp.symbols("blockIdx.x blockIdx.y blockIdx.z"))
THREAD_IDX = list(sp.symbols("threadIdx.x threadIdx.y threadIdx.z"))


def getLinewiseCoordinates(field, ghostLayers):
    availableIndices = [THREAD_IDX[0]] + BLOCK_IDX
    assert field.spatialDimensions <= 4, "This indexing scheme supports at most 4 spatial dimensions"
    result = availableIndices[:field.spatialDimensions]

    fastestCoordinate = field.layout[-1]
    result[0], result[fastestCoordinate] = result[fastestCoordinate], result[0]

    def getCallParameters(arrShape):
        def getShapeOfCudaIdx(cudaIdx):
            if cudaIdx not in result:
                return 1
            else:
                return arrShape[result.index(cudaIdx)] - 2 * ghostLayers

        return {'block': tuple([getShapeOfCudaIdx(idx) for idx in THREAD_IDX]),
                'grid': tuple([getShapeOfCudaIdx(idx) for idx in BLOCK_IDX])}

    return [i + ghostLayers for i in result], getCallParameters


def createCUDAKernel(listOfEquations, functionName="kernel", typeForSymbol=None):
    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    allFields = fieldsRead.union(fieldsWritten)
    readOnlyFields = set([f.name for f in fieldsRead - fieldsWritten])

    ast = KernelFunction(Block(assignments), allFields, functionName)
    ast.globalVariables.update(BLOCK_IDX + THREAD_IDX)

    fieldAccesses = ast.atoms(Field.Access)
    requiredGhostLayers = max([fa.requiredGhostLayers for fa in fieldAccesses])

    coordMapping, getCallParameters = getLinewiseCoordinates(list(fieldsRead)[0], requiredGhostLayers)
    basePointerInfo = [['spatialInner0']]
    basePointerInfos = {f.name: parseBasePointerInfo(basePointerInfo, [2, 1, 0], f) for f in allFields}

    coordMapping = {f.name: coordMapping for f in allFields}
    resolveFieldAccesses(ast, readOnlyFields, fieldToFixedCoordinates=coordMapping,
                         fieldToBasePointerInfo=basePointerInfos)
    # add the function which determines #blocks and #threads as additional member to KernelFunction node
    # this is used by the jit
    ast.getCallParameters = getCallParameters
    return ast


def createdIndexedCUDAKernel(listOfEquations, indexFields, functionName="kernel", typeForSymbol=None,
                             coordinateNames=('x', 'y', 'z')):
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

    functionBody = Block(coordinateSymbolAssignments + assignments)
    ast = KernelFunction(functionBody, allFields, functionName)
    ast.globalVariables.update(BLOCK_IDX + THREAD_IDX)

    coordMapping, getCallParameters = getLinewiseCoordinates(list(indexFields)[0], ghostLayers=0)
    basePointerInfo = [['spatialInner0']]
    basePointerInfos = {f.name: parseBasePointerInfo(basePointerInfo, [2, 1, 0], f) for f in allFields}

    coordMapping = {f.name: coordMapping for f in indexFields}
    coordMapping.update({f.name: coordinateTypedSymbols for f in nonIndexFields})
    resolveFieldAccesses(ast, readOnlyFields, fieldToFixedCoordinates=coordMapping,
                         fieldToBasePointerInfo=basePointerInfos)
    # add the function which determines #blocks and #threads as additional member to KernelFunction node
    # this is used by the jit
    ast.getCallParameters = getCallParameters
    return ast