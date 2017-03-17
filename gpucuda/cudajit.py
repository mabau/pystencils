import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pystencils.backends.cbackend import generateC
from pystencils.transformations import symbolNameToVariableName
from pystencils.types import StructType


def makePythonFunction(kernelFunctionNode, argumentDict={}):
    """
    Creates a kernel function from an abstract syntax tree which
    was created e.g. by :func:`pystencils.gpucuda.createCUDAKernel`
    or :func:`pystencils.gpucuda.createdIndexedCUDAKernel`

    :param kernelFunctionNode: the abstract syntax tree
    :param argumentDict: parameters passed here are already fixed. Remaining parameters have to be passed to the
                        returned kernel functor.
    :return: kernel functor
    """
    code = "#include <cstdint>\n"
    code += "#define FUNC_PREFIX __global__\n"
    code += "#define RESTRICT __restrict__\n\n"
    code += str(generateC(kernelFunctionNode))

    mod = SourceModule(code, options=["-w", "-std=c++11"])
    func = mod.get_function(kernelFunctionNode.functionName)

    def wrapper(**kwargs):
        from copy import copy
        fullArguments = copy(argumentDict)
        fullArguments.update(kwargs)
        shape = _checkArguments(kernelFunctionNode.parameters, fullArguments)

        dictWithBlockAndThreadNumbers = kernelFunctionNode.getCallParameters(shape)

        args = _buildNumpyArgumentList(kernelFunctionNode, fullArguments)
        func(*args, **dictWithBlockAndThreadNumbers)
    return wrapper


def _buildNumpyArgumentList(kernelFunctionNode, argumentDict):
    argumentDict = {symbolNameToVariableName(k): v for k, v in argumentDict.items()}
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
            expectedType = arg.dtype.numpyDtype
            result.append(expectedType(param))
    return result


def _checkArguments(parameterSpecification, argumentDict):
    """
    Checks if parameters passed to kernel match the description in the AST function node.
    If not it raises a ValueError, on success it returns the array shape that determines the CUDA blocks and threads
    """
    argumentDict = {symbolNameToVariableName(k): v for k, v in argumentDict.items()}
    arrayShapes = set()
    indexArrShapes = set()
    for arg in parameterSpecification:
        if arg.isFieldArgument:
            try:
                fieldArr = argumentDict[arg.fieldName]
            except KeyError:
                raise KeyError("Missing field parameter for kernel call " + arg.fieldName)

            symbolicField = arg.field
            if arg.isFieldPtrArgument:
                if symbolicField.hasFixedShape:
                    symbolicFieldShape = tuple(int(i) for i in symbolicField.shape)
                    if isinstance(symbolicField.dtype, StructType):
                        symbolicFieldShape = symbolicFieldShape[:-1]
                    if symbolicFieldShape != fieldArr.shape:
                        raise ValueError("Passed array '%s' has shape %s which does not match expected shape %s" %
                                         (arg.fieldName, str(fieldArr.shape), str(symbolicField.shape)))
                if symbolicField.hasFixedShape:
                    symbolicFieldStrides = tuple(int(i) * fieldArr.dtype.itemsize for i in symbolicField.strides)
                    if isinstance(symbolicField.dtype, StructType):
                        symbolicFieldStrides = symbolicFieldStrides[:-1]
                    if symbolicFieldStrides != fieldArr.strides:
                        raise ValueError("Passed array '%s' has strides %s which does not match expected strides %s" %
                                         (arg.fieldName, str(fieldArr.strides), str(symbolicFieldStrides)))

                if symbolicField.isIndexField:
                    indexArrShapes.add(fieldArr.shape[:symbolicField.spatialDimensions])
                else:
                    arrayShapes.add(fieldArr.shape[:symbolicField.spatialDimensions])

    if len(arrayShapes) > 1:
        raise ValueError("All passed arrays have to have the same size " + str(arrayShapes))
    if len(indexArrShapes) > 1:
        raise ValueError("All passed index arrays have to have the same size " + str(arrayShapes))

    if len(indexArrShapes) > 0:
        return list(indexArrShapes)[0]
    else:
        return list(arrayShapes)[0]



