from collections import defaultdict

import sympy as sp

from pystencils.generator import resolveFieldAccesses
from pystencils.generator import typeAllEquations, Block, KernelFunction, parseBasePointerInfo

BLOCK_IDX = list(sp.symbols("blockIdx.x blockIdx.y blockIdx.z"))
THREAD_IDX = list(sp.symbols("threadIdx.x threadIdx.y threadIdx.z"))

"""
GPU Access Patterns

- knows about the iteration range
- know about mapping of field indices to CUDA block and thread indices
- iterates over spatial coordinates - constructed with a specific number of coordinates
- can
"""


def getLinewiseCoordinateAccessExpression(field, indexCoordinate):
    availableIndices = [THREAD_IDX[0]] + BLOCK_IDX
    fastestCoordinate = field.layout[-1]
    availableIndices[fastestCoordinate], availableIndices[0] = availableIndices[0], availableIndices[fastestCoordinate]
    cudaIndices = availableIndices[:field.spatialDimensions]

    offsetToCell = sum([cudaIdx * stride for cudaIdx, stride in zip(cudaIndices, field.spatialStrides)])
    indexOffset = sum([idx * indexStride for idx, indexStride in zip(indexCoordinate, field.indexStrides)])
    return sp.simplify(offsetToCell + indexOffset)


def getLinewiseCoordinates(field):
    availableIndices = [THREAD_IDX[0]] + BLOCK_IDX
    d = field.spatialDimensions + field.indexDimensions
    fastestCoordinate = field.layout[-1]
    result = availableIndices[:d]
    result[0], result[fastestCoordinate] = result[fastestCoordinate], result[0]
    return result


def createCUDAKernel(listOfEquations, functionName="kernel", typeForSymbol=defaultdict(lambda: "double")):
    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    for f in fieldsRead - fieldsWritten:
        f.setReadOnly()

    code = KernelFunction(Block(assignments), functionName)
    code.qualifierPrefix = "__global__ "
    code.variablesToIgnore.update(BLOCK_IDX + THREAD_IDX)

    coordMapping = getLinewiseCoordinates(list(fieldsRead)[0])
    allFields = fieldsRead.union(fieldsWritten)
    basePointerInfo = [['spatialInner0']]
    basePointerInfos = {f.name: parseBasePointerInfo(basePointerInfo, [0, 1, 2], f) for f in allFields}
    resolveFieldAccesses(code, fieldToFixedCoordinates={'src': coordMapping, 'dst': coordMapping},
                         fieldToBasePointerInfo=basePointerInfos)
    return code


if __name__ == "__main__":
    import sympy as sp
    from lbmpy.stencils import getStencil
    from lbmpy.collisionoperator import makeSRT
    from lbmpy.lbmgenerator import createLbmEquations

    latticeModel = makeSRT(getStencil("D2Q9"), order=2, compressible=False)
    r = createLbmEquations(latticeModel, doCSE=True)
    kernel = createCUDAKernel(r)
    print(kernel.generateC())

    from pycuda.compiler import SourceModule

    mod = SourceModule(str(kernel.generateC()))
    func = mod.get_function("kernel")
