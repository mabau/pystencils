import os
import subprocess
from ctypes import cdll, c_double, c_float, sizeof
from tempfile import TemporaryDirectory
from pystencils.backends.cbackend import printCCode
import numpy as np


CONFIG_GCC = {
    'compiler': 'g++',
    'flags': '-Ofast -DNDEBUG -fPIC -shared -march=native -fopenmp',
}
CONFIG_INTEL = {
    'compiler': '/software/intel/2017/bin/icpc',
    'flags': '-Ofast -DNDEBUG -fPIC -shared -march=native -fopenmp -Wl,-rpath=/software/intel/2017/lib/intel64',
    'env': {
        'INTEL_LICENSE_FILE': '1713@license4.rrze.uni-erlangen.de',
        'LM_PROJECT': 'iwia',
    }
}
CONFIG_CLANG = {
    'compiler': 'clang++',
    'flags': '-Ofast -DNDEBUG -fPIC -shared -march=native -fopenmp',
}
CONFIG = CONFIG_INTEL


def ctypeFromString(typename, includePointers=True):
    import ctypes as ct

    typename = typename.replace("*", " * ")
    typeComponents = typename.split()

    basicTypeMap = {
        'double': ct.c_double,
        'float': ct.c_float,
        'int': ct.c_int,
        'long': ct.c_long,
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


def ctypeFromNumpyType(numpyType):
    typeMap = {
        np.dtype('float64'): c_double,
        np.dtype('float32'): c_float,
    }
    return typeMap[numpyType]


def compileAndLoad(kernelFunctionNode):
    with TemporaryDirectory() as tmpDir:
        srcFile = os.path.join(tmpDir, 'source.cpp')
        with open(srcFile, 'w') as sourceFile:
            print('#include <iostream>', file=sourceFile)
            print("#include <cmath>", file=sourceFile)
            print('extern "C" { ', file=sourceFile)
            print(printCCode(kernelFunctionNode), file=sourceFile)
            print('}', file=sourceFile)

        compilerCmd = [CONFIG['compiler']] + CONFIG['flags'].split()
        libFile = os.path.join(tmpDir, "jit.so")
        compilerCmd += [srcFile, '-o', libFile]
        configEnv = CONFIG['env'] if 'env' in CONFIG else {}
        env = os.environ.copy()
        env.update(configEnv)
        subprocess.call(compilerCmd, env=env)

        showAssembly = False
        if showAssembly:
            assemblyFile = os.path.join(tmpDir, "assembly.s")
            compilerCmd = [CONFIG['compiler'], '-S', '-o', assemblyFile, srcFile] + CONFIG['flags'].split()
            subprocess.call(compilerCmd, env=env)
            assembly = open(assemblyFile, 'r').read()
            kernelFunctionNode.assembly = assembly
        loadedJitLib = cdll.LoadLibrary(libFile)

    return loadedJitLib


def buildCTypeArgumentList(kernelFunctionNode, argumentDict):
    ctArguments = []
    for arg in kernelFunctionNode.parameters:
        if arg.isFieldArgument:
            field = argumentDict[arg.fieldName]
            if arg.isFieldPtrArgument:
                ctArguments.append(field.ctypes.data_as(ctypeFromString(arg.dtype)))
            elif arg.isFieldShapeArgument:
                dataType = ctypeFromString(arg.dtype, includePointers=False)
                ctArguments.append(field.ctypes.shape_as(dataType))
            elif arg.isFieldStrideArgument:
                dataType = ctypeFromString(arg.dtype, includePointers=False)
                baseFieldType = ctypeFromNumpyType(field.dtype)
                strides = field.ctypes.strides_as(dataType)
                for i in range(len(field.shape)):
                    assert strides[i] % sizeof(baseFieldType) == 0
                    strides[i] //= sizeof(baseFieldType)
                ctArguments.append(strides)
            else:
                assert False
        else:
            param = argumentDict[arg.name]
            expectedType = ctypeFromString(arg.dtype)
            ctArguments.append(expectedType(param))
    return ctArguments


def makePythonFunctionIncompleteParams(kernelFunctionNode, argumentDict):
    func = compileAndLoad(kernelFunctionNode)[kernelFunctionNode.functionName]
    func.restype = None

    def wrapper(**kwargs):
        from copy import copy
        fullArguments = copy(argumentDict)
        fullArguments.update(kwargs)
        args = buildCTypeArgumentList(kernelFunctionNode, fullArguments)
        func(*args)
    return wrapper


def makePythonFunction(kernelFunctionNode, argumentDict={}):
    # build up list of CType arguments
    try:
        args = buildCTypeArgumentList(kernelFunctionNode, argumentDict)
    except KeyError:
        # not all parameters specified yet
        return makePythonFunctionIncompleteParams(kernelFunctionNode, argumentDict)
    func = compileAndLoad(kernelFunctionNode)[kernelFunctionNode.functionName]
    func.restype = None
    return lambda: func(*args)







