from functools import partial

from pystencils.gpucuda.indexing import BlockIndexing
from pystencils.transformations import resolveFieldAccesses, typeAllEquations, parseBasePointerInfo, getCommonShape, \
    substituteArrayAccessesWithConstants, resolveBufferAccesses
from pystencils.astnodes import Block, KernelFunction, SympyAssignment, LoopOverCoordinate
from pystencils.data_types import TypedSymbol, BasicType, StructType
from pystencils import Field, FieldType
from pystencils.gpucuda.cudajit import makePythonFunction


def createCUDAKernel(listOfEquations, functionName="kernel", typeForSymbol=None, indexingCreator=BlockIndexing,
                     iterationSlice=None, ghostLayers=None):
    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    allFields = fieldsRead.union(fieldsWritten)
    readOnlyFields = set([f.name for f in fieldsRead - fieldsWritten])

    buffers = set([f for f in allFields if FieldType.isBuffer(f)])
    fieldsWithoutBuffers = allFields - buffers

    fieldAccesses = set()
    numBufferAccesses = 0
    for eq in listOfEquations:
        fieldAccesses.update(eq.atoms(Field.Access))

        numBufferAccesses += sum([1 for access in eq.atoms(Field.Access) if FieldType.isBuffer(access.field)])

    commonShape = getCommonShape(fieldsWithoutBuffers)

    if iterationSlice is None:
        # determine iteration slice from ghost layers
        if ghostLayers is None:
            # determine required number of ghost layers from field access
            requiredGhostLayers = max([fa.requiredGhostLayers for fa in fieldAccesses])
            ghostLayers = [(requiredGhostLayers, requiredGhostLayers)] * len(commonShape)
        iterationSlice = []
        if isinstance(ghostLayers, int):
            for i in range(len(commonShape)):
                iterationSlice.append(slice(ghostLayers, -ghostLayers if ghostLayers > 0 else None))
        else:
            for i in range(len(commonShape)):
                iterationSlice.append(slice(ghostLayers[i][0], -ghostLayers[i][1] if ghostLayers[i][1] > 0 else None))

    indexing = indexingCreator(field=list(fieldsWithoutBuffers)[0], iterationSlice=iterationSlice)

    block = Block(assignments)
    block = indexing.guard(block, commonShape)
    ast = KernelFunction(block, functionName=functionName, ghostLayers=ghostLayers, backend='gpucuda')
    ast.globalVariables.update(indexing.indexVariables)

    coordMapping = indexing.coordinates
    basePointerInfo = [['spatialInner0']]
    basePointerInfos = {f.name: parseBasePointerInfo(basePointerInfo, [2, 1, 0], f) for f in allFields}

    coordMapping = {f.name: coordMapping for f in allFields}

    loopVars = [numBufferAccesses * i for i in indexing.coordinates]
    loopStrides = list(fieldsWithoutBuffers)[0].shape

    baseBufferIndex = loopVars[0]
    stride = 1
    for idx, var in enumerate(loopVars[1:]):
        stride *= loopStrides[idx]
        baseBufferIndex += var * stride

    resolveBufferAccesses(ast, baseBufferIndex, readOnlyFields)
    resolveFieldAccesses(ast, readOnlyFields, fieldToBasePointerInfo=basePointerInfos,
                         fieldToFixedCoordinates=coordMapping)

    substituteArrayAccessesWithConstants(ast)

    # add the function which determines #blocks and #threads as additional member to KernelFunction node
    # this is used by the jit

    # If loop counter symbols have been explicitly used in the update equations (e.g. for built in periodicity),
    # they are defined here
    undefinedLoopCounters = {LoopOverCoordinate.isLoopCounterSymbol(s): s for s in ast.body.undefinedSymbols
                             if LoopOverCoordinate.isLoopCounterSymbol(s) is not None}
    for i, loopCounter in undefinedLoopCounters.items():
        ast.body.insertFront(SympyAssignment(loopCounter, indexing.coordinates[i]))

    ast.indexing = indexing
    ast.compile = partial(makePythonFunction, ast)
    return ast


def createdIndexedCUDAKernel(listOfEquations, indexFields, functionName="kernel", typeForSymbol=None,
                             coordinateNames=('x', 'y', 'z'), indexingCreator=BlockIndexing):
    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    allFields = fieldsRead.union(fieldsWritten)
    readOnlyFields = set([f.name for f in fieldsRead - fieldsWritten])

    for indexField in indexFields:
        indexField.fieldType = FieldType.INDEXED
        assert FieldType.isIndexed(indexField)
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
    indexing = indexingCreator(field=idxField, iterationSlice=[slice(None, None, None)] * len(idxField.spatialShape))

    functionBody = Block(coordinateSymbolAssignments + assignments)
    functionBody = indexing.guard(functionBody, getCommonShape(indexFields))
    ast = KernelFunction(functionBody, functionName=functionName, backend='gpucuda')
    ast.globalVariables.update(indexing.indexVariables)

    coordMapping = indexing.coordinates
    basePointerInfo = [['spatialInner0']]
    basePointerInfos = {f.name: parseBasePointerInfo(basePointerInfo, [2, 1, 0], f) for f in allFields}

    coordMapping = {f.name: coordMapping for f in indexFields}
    coordMapping.update({f.name: coordinateTypedSymbols for f in nonIndexFields})
    resolveFieldAccesses(ast, readOnlyFields, fieldToFixedCoordinates=coordMapping,
                         fieldToBasePointerInfo=basePointerInfos)
    substituteArrayAccessesWithConstants(ast)

    # add the function which determines #blocks and #threads as additional member to KernelFunction node
    # this is used by the jit
    ast.indexing = indexing
    ast.compile = partial(makePythonFunction, ast)
    return ast
