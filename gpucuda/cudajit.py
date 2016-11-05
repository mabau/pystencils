import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def numpyTypeFromString(typename, includePointers=True):
    import ctypes as ct

    typename = typename.replace("*", " * ")
    typeComponents = typename.split()

    basicTypeMap = {
        'double': np.float64,
        'float': np.float32,
        'int': np.int32,
        'long': np.int64,
    }

    resultType = None
    for typeComponent in typeComponents:
        typeComponent = typeComponent.strip()
        if typeComponent == "const" or typeComponent == "restrict" or typeComponent == "volatile":
            continue
        if typeComponent in basicTypeMap:
            resultType = basicTypeMap[typeComponent]
        elif typeComponent == "*" and includePointers:
            assert resultType is not None
            resultType = ct.POINTER(resultType)

    return resultType


def buildNumpyArgumentList(kernelFunctionNode, argumentDict):
    result = []
    for arg in kernelFunctionNode.parameters:
        if arg.isFieldArgument:
            field = argumentDict[arg.fieldName]
            if arg.isFieldPtrArgument:
                result.append(field.gpudata)
            elif arg.isFieldShapeArgument:
                strideArr = np.array(field.strides, dtype=np.int32) / field.dtype.itemsize
                result.append(cuda.In(strideArr))
            elif arg.isFieldStrideArgument:
                shapeArr = np.array(field.shape, dtype=np.int32)
                result.append(cuda.In(shapeArr))
            else:
                assert False
        else:
            param = argumentDict[arg.name]
            expectedType = numpyTypeFromString(arg.dtype)
            result.append(expectedType(param))
    return result


def makePythonFunction(kernelFunctionNode, argumentDict={}):
    mod = SourceModule(str(kernelFunctionNode.generateC()))
    func = mod.get_function(kernelFunctionNode.functionName)

    # 1) get argument list
    args = buildNumpyArgumentList(kernelFunctionNode, argumentDict)

    # 2) determine block and grid tuples
    dictWithBlockAndThreadNumbers = kernelFunctionNode.getCallParameters()
    
    # TODO prepare the function here

