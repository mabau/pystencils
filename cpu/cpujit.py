from __future__ import print_function
import os
import subprocess
from ctypes import cdll, c_double, c_float, sizeof
import tempfile
import shutil
from pystencils.backends.cbackend import generateC
import numpy as np
import pickle
import hashlib

from pystencils.transformations import symbolNameToVariableName

CONFIG_GCC = {
    'compiler': 'g++-4.8',
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
CONFIG_INTEL_SUPERMUC = {
    'compiler': '/lrz/sys/intel/studio2017_u1/compilers_and_libraries_2017.1.132/linux/bin/intel64/icpc',
    'flags': '-Ofast -DNDEBUG -fPIC -shared -march=native -fopenmp -Wl,'
             '-rpath=/lrz/sys/intel/studio2016_u4/compilers_and_libraries_2016.4.258/linux/mkl/lib/intel64',
    'env': {
        'INTEL_LICENSE_FILE': '/lrz/sys/intel/licenses',
    }
}
CONFIG_CLANG = {
    'compiler': 'clang++',
    'flags': '-Ofast -DNDEBUG -fPIC -shared -march=native ',
}
CONFIG = CONFIG_GCC


if CONFIG is CONFIG_CLANG and not 'Apple LLVM' in subprocess.check_output(['clang++', '--version']):
    CONFIG_CLANG['flags'] += '-fopenmp'


def ctypeFromString(typename, includePointers=True):
    import ctypes as ct

    typename = str(typename).replace("*", " * ")
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


def compile(code, tmpDir, libFile, createAssemblyCode=False):
    srcFile = os.path.join(tmpDir, 'source.cpp')
    with open(srcFile, 'w') as sourceFile:
        print('#include <iostream>', file=sourceFile)
        print("#include <cmath>", file=sourceFile)
        print('extern "C" { ', file=sourceFile)
        print(code, file=sourceFile)
        print('}', file=sourceFile)

    compilerCmd = [CONFIG['compiler']] + CONFIG['flags'].split()
    compilerCmd += [srcFile, '-o', libFile]
    configEnv = CONFIG['env'] if 'env' in CONFIG else {}
    env = os.environ.copy()
    env.update(configEnv)
    try:
        subprocess.check_output(compilerCmd, env=env, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise e

    assembly = None
    if createAssemblyCode:
        assemblyFile = os.path.join(tmpDir, "assembly.s")
        compilerCmd = [CONFIG['compiler'], '-S', '-o', assemblyFile, srcFile] + CONFIG['flags'].split()
        subprocess.call(compilerCmd, env=env)
        assembly = open(assemblyFile, 'r').read()
    return assembly


def compileAndLoad(kernelFunctionNode):
    tmpDir = tempfile.mkdtemp()
    libFile = os.path.join(tmpDir, "jit.so")
    compile(generateC(kernelFunctionNode), tmpDir, libFile)
    loadedJitLib = cdll.LoadLibrary(libFile)
    shutil.rmtree(tmpDir)

    return loadedJitLib


def buildCTypeArgumentList(parameterSpecification, argumentDict):
    argumentDict = {symbolNameToVariableName(k): v for k, v in argumentDict.items()}
    ctArguments = []
    arrayShapes = set()
    for arg in parameterSpecification:
        if arg.isFieldArgument:
            field = argumentDict[arg.fieldName]
            symbolicField = arg.field
            if arg.isFieldPtrArgument:
                ctArguments.append(field.ctypes.data_as(ctypeFromString(arg.dtype)))
                if symbolicField.hasFixedShape:
                    if tuple(int(i) for i in symbolicField.shape) != field.shape:
                        raise ValueError("Passed array '%s' has shape %s which does not match expected shape %s" %
                                         (arg.fieldName, str(field.shape), str(symbolicField.shape)))
                if symbolicField.hasFixedShape:
                    if tuple(int(i) * field.itemsize for i in symbolicField.strides) != field.strides:
                        raise ValueError("Passed array '%s' has strides %s which does not match expected strides %s" %
                                         (arg.fieldName, str(field.strides), str(symbolicField.strides)))

                if not symbolicField.isIndexField:
                    arrayShapes.add(field.shape[:symbolicField.spatialDimensions])
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

    if len(arrayShapes) > 1:
        raise ValueError("All passed arrays have to have the same size " + str(arrayShapes))
    return ctArguments


def makePythonFunctionIncompleteParams(kernelFunctionNode, argumentDict):
    func = compileAndLoad(kernelFunctionNode)[kernelFunctionNode.functionName]
    func.restype = None
    parameters = kernelFunctionNode.parameters

    def wrapper(**kwargs):
        from copy import copy
        fullArguments = copy(argumentDict)
        fullArguments.update(kwargs)
        args = buildCTypeArgumentList(parameters, fullArguments)
        func(*args)
    return wrapper


def makePythonFunction(kernelFunctionNode, argumentDict={}):
    """
    Creates C code from the abstract syntax tree, compiles it and makes it accessible as Python function

    The parameters of the kernel are:
        - numpy arrays for each field used in the kernel. The keyword argument name is the name of the field
        - all symbols which are not defined in the kernel itself are expected as parameters

    :param kernelFunctionNode: the abstract syntax tree
    :param argumentDict: parameters passed here are already fixed. Remaining parameters have to be passed to the
                        returned kernel functor.
    :return: kernel functor
    """
    # build up list of CType arguments
    try:
        args = buildCTypeArgumentList(kernelFunctionNode.parameters, argumentDict)
    except KeyError:
        # not all parameters specified yet
        return makePythonFunctionIncompleteParams(kernelFunctionNode, argumentDict)
    func = compileAndLoad(kernelFunctionNode)[kernelFunctionNode.functionName]
    func.restype = None
    return lambda: func(*args)


class CachedKernel(object):
    def __init__(self, configDict, ast, parameterValues):
        self.configDict = configDict
        self.ast = ast
        self.parameterValues = parameterValues
        self.funcPtr = None

    def __compile(self):
        self.funcPtr = makePythonFunction(self.ast, self.parameterValues)

    def __call__(self, *args, **kwargs):
        if self.funcPtr is None:
            self.__compile()
        self.funcPtr(*args, **kwargs)


def hashToFunctionName(h):
    res = "func_%s" % (h,)
    return res.replace('-', 'm')


def createLibrary(cachedKernels, libraryFile):
    libraryInfoFile = libraryFile + ".info"

    tmpDir = tempfile.mkdtemp()
    code = ""
    infoDict = {}
    for cachedKernel in cachedKernels:
        s = repr(sorted(cachedKernel.configDict.items()))
        configHash = hashlib.sha1(s.encode()).hexdigest()
        cachedKernel.ast.functionName = hashToFunctionName(configHash)
        kernelCode = generateC(cachedKernel.ast)
        code += kernelCode + "\n"
        infoDict[configHash] = {'code': kernelCode,
                                'parameterValues': cachedKernel.parameterValues,
                                'configDict': cachedKernel.configDict,
                                'parameterSpecification': cachedKernel.ast.parameters}

    compile(code, tmpDir, libraryFile)
    pickle.dump(infoDict, open(libraryInfoFile, "wb"))
    shutil.rmtree(tmpDir)


def loadLibrary(libraryFile):
    libraryInfoFile = libraryFile + ".info"

    libraryFile = cdll.LoadLibrary(libraryFile)
    libraryInfo = pickle.load(open(libraryInfoFile, 'rb'))

    def getKernel(**kwargs):
        s = repr(sorted(kwargs.items()))
        configHash = hashlib.sha1(s.encode()).hexdigest()
        if configHash not in libraryInfo:
            raise ValueError("No such kernel in library")
        func = libraryFile[hashToFunctionName(configHash)]
        func.restype = None

        def wrapper(**kwargs):
            from copy import copy
            fullArguments = copy(libraryInfo[configHash]['parameterValues'])
            fullArguments.update(kwargs)
            args = buildCTypeArgumentList(libraryInfo[configHash]['parameterSpecification'], fullArguments)
            func(*args)
        wrapper.configDict = libraryInfo[configHash]['configDict']
        return wrapper

    return getKernel
