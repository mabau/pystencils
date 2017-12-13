import numpy as np
from pystencils.backends.cbackend import generateC
from pystencils.transformations import symbolNameToVariableName
from pystencils.data_types import StructType, getBaseType
from pystencils.field import FieldType


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
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    code = "#include <cstdint>\n"
    code += "#define FUNC_PREFIX __global__\n"
    code += "#define RESTRICT __restrict__\n\n"
    code += str(generateC(kernelFunctionNode))

    mod = SourceModule(code, options=["-w", "-std=c++11"])
    func = mod.get_function(kernelFunctionNode.functionName)

    parameters = kernelFunctionNode.parameters

    cache = {}
    cacheValues = []

    def wrapper(**kwargs):
        key = hash(tuple((k, id(v)) for k, v in kwargs.items()))
        try:
            args, dictWithBlockAndThreadNumbers = cache[key]
            func(*args, **dictWithBlockAndThreadNumbers)
        except KeyError:
            fullArguments = argumentDict.copy()
            fullArguments.update(kwargs)
            shape = _checkArguments(parameters, fullArguments)

            indexing = kernelFunctionNode.indexing
            dictWithBlockAndThreadNumbers = indexing.getCallParameters(shape)
            dictWithBlockAndThreadNumbers['block'] = tuple(int(i) for i in dictWithBlockAndThreadNumbers['block'])
            dictWithBlockAndThreadNumbers['grid'] = tuple(int(i) for i in dictWithBlockAndThreadNumbers['grid'])

            args = _buildNumpyArgumentList(parameters, fullArguments)
            cache[key] = (args, dictWithBlockAndThreadNumbers)
            cacheValues.append(kwargs)  # keep objects alive such that ids remain unique
            func(*args, **dictWithBlockAndThreadNumbers)
        #cuda.Context.synchronize() # useful for debugging, to get errors right after kernel was called
    return wrapper


def _buildNumpyArgumentList(parameters, argumentDict):
    import pycuda.driver as cuda

    argumentDict = {symbolNameToVariableName(k): v for k, v in argumentDict.items()}
    result = []
    for arg in parameters:
        if arg.isFieldArgument:
            field = argumentDict[arg.fieldName]
            if arg.isFieldPtrArgument:
                actualType = field.dtype
                expectedType = arg.dtype.baseType.numpyDtype
                if expectedType != actualType:
                    raise ValueError("Data type mismatch for field '%s'. Expected '%s' got '%s'." %
                                     (arg.fieldName, expectedType, actualType))
                result.append(field)
            elif arg.isFieldStrideArgument:
                dtype = getBaseType(arg.dtype).numpyDtype
                strideArr = np.array(field.strides, dtype=dtype) // field.dtype.itemsize
                result.append(cuda.In(strideArr))
            elif arg.isFieldShapeArgument:
                dtype = getBaseType(arg.dtype).numpyDtype
                shapeArr = np.array(field.shape, dtype=dtype)
                result.append(cuda.In(shapeArr))
            else:
                assert False
        else:
            param = argumentDict[arg.name]
            expectedType = arg.dtype.numpyDtype
            result.append(expectedType.type(param))
    assert len(result) == len(parameters)
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

                if FieldType.isIndexed(symbolicField):
                    indexArrShapes.add(fieldArr.shape[:symbolicField.spatialDimensions])
                elif not FieldType.isBuffer(symbolicField):
                    arrayShapes.add(fieldArr.shape[:symbolicField.spatialDimensions])

    if len(arrayShapes) > 1:
        raise ValueError("All passed arrays have to have the same size " + str(arrayShapes))
    if len(indexArrShapes) > 1:
        raise ValueError("All passed index arrays have to have the same size " + str(arrayShapes))

    if len(indexArrShapes) > 0:
        return list(indexArrShapes)[0]
    else:
        return list(arrayShapes)[0]



