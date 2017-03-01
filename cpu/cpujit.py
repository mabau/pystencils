from __future__ import print_function
import os
import subprocess
from ctypes import cdll, c_double, c_float, sizeof
import shutil
from pystencils.backends.cbackend import generateC
import numpy as np
import hashlib
import json
import platform
import glob
import atexit
from collections import OrderedDict, Mapping
from pystencils.transformations import symbolNameToVariableName


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
    func = compileAndLoad(kernelFunctionNode)
    func.restype = None
    return lambda: func(*args)


def setCompilerConfig(config):
    """
    Override the configuration provided in config file

    Configuration of compiler parameters:
    If this function is not called the configuration is taken from a config file in JSON format which
    is searched in the following locations in the order specified:
        - at location provided in environment variable PYSTENCILS_CONFIG (if this variable exists)
        - a file called ".pystencils.json" in the current working directory
        - ~/.pystencils.json in your home
    If none of these files exist a file ~/.pystencils.json is created with a default configuration using
    the GNU 'g++'

    An example JSON file with all possible keys. If not all keys are specified, default values are used
    ``
    {
        'compiler' :
        {
            "command": "/software/intel/2017/bin/icpc",
            "flags": "-Ofast -DNDEBUG -fPIC -march=native -fopenmp",
            "env": {
                "LM_PROJECT": "iwia",
            }
        }
    }
    ``
    """
    global _config
    _config = config.copy()


def _recursiveDictUpdate(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = _recursiveDictUpdate(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def getConfigurationFilePath():
    if platform.system().lower() == 'linux':
        configPathInHome = os.path.expanduser(os.path.join("~", '.config', 'pystencils', 'config.json'))
    else:
        configPathInHome = os.path.expanduser(os.path.join("~", '.pystencils', 'config.json'))

    # 1) Read path from environment variable if found
    if 'PYSTENCILS_CONFIG' in os.environ:
        return os.environ['PYSTENCILS_CONFIG'], True
    # 2) Look in current directory for pystencils.json
    elif os.path.exists("pystencils.json"):
        return "pystencils.json", True
    # 3) Try ~/.pystencils.json
    elif os.path.exists(configPathInHome):
        return configPathInHome, True
    else:
        return configPathInHome, False


def createFolder(path, isFile):
    if isFile:
        path = os.path.split(path)[0]
    try:
        os.makedirs(path)
    except os.error:
        pass


def readConfig():
    if platform.system().lower() == 'linux':
        defaultCompilerConfig = OrderedDict([
            ('os', 'linux'),
            ('command', 'g++'),
            ('flags', '-Ofast -DNDEBUG -fPIC -march=native -fopenmp'),
            ('restrictQualifier', '__restrict__')
        ])
        defaultCacheConfig = OrderedDict([
            ('readFromSharedLibrary', False),
            ('objectCache', '/tmp/pystencils/objectcache'),
            ('clearCacheOnStart', False),
            ('sharedLibrary', '/tmp/pystencils/cache.so'),
        ])
    elif platform.system().lower() == 'windows':
        defaultCompilerConfig = OrderedDict([
            ('os', 'windows'),
            ('msvcVersion', 'latest'),
            ('arch', 'x64'),
            ('flags', '/Ox /fp:fast /openmp /arch:avx'),
            ('restrictQualifier', '__restrict')
        ])
        defaultCacheConfig = OrderedDict([
            ('readFromSharedLibrary', False),
            ('objectCache', os.path.join('~', '.pystencils', 'objectcache')),
            ('clearCacheOnStart', False),
            ('sharedLibrary', os.path.join('~', '.pystencils', 'cache.dll')),
        ])

    defaultConfig = OrderedDict([('compiler', defaultCompilerConfig),
                                 ('cache', defaultCacheConfig)])

    configPath, configExists = getConfigurationFilePath()
    config = defaultConfig.copy()
    if configExists:
        loadedConfig = json.load(open(configPath, 'r'))
        config = _recursiveDictUpdate(config, loadedConfig)
    else:
        createFolder(configPath, True)
        json.dump(config, open(configPath, 'w'), indent=4)

    config['cache']['sharedLibrary'] = os.path.expanduser(config['cache']['sharedLibrary'])
    config['cache']['objectCache'] = os.path.expanduser(config['cache']['objectCache'])

    if config['cache']['clearCacheOnStart']:
        shutil.rmtree(config['cache']['objectCache'], ignore_errors=True)

    createFolder(config['cache']['objectCache'], False)
    createFolder(config['cache']['sharedLibrary'], True)

    if 'env' not in config['compiler']:
        config['compiler']['env'] = {}

    if config['compiler']['os'] == 'windows':
        from pystencils.cpu.msvc_detection import getEnvironment
        msvcEnv = getEnvironment(config['compiler']['msvcVersion'], config['compiler']['arch'])
        config['compiler']['env'].update(msvcEnv)

    return config


_config = readConfig()


def getCompilerConfig():
    return _config['compiler']


def getCacheConfig():
    return _config['cache']


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


def hashToFunctionName(h):
    res = "func_%s" % (h,)
    return res.replace('-', 'm')


def compileObjectCacheToSharedLibrary():
    compilerConfig = getCompilerConfig()
    cacheConfig = getCacheConfig()

    sharedLibrary = cacheConfig['sharedLibrary']
    if len(sharedLibrary) == 0 or cacheConfig['readFromSharedLibrary']:
        return

    configEnv = compilerConfig['env'] if 'env' in compilerConfig else {}
    compileEnvironment = os.environ.copy()
    compileEnvironment.update(configEnv)

    try:
        if compilerConfig['os'] == 'windows':
            allObjectFiles = glob.glob(os.path.join(cacheConfig['objectCache'], '*.obj'))
            linkCmd = ['link.exe',  '/DLL', '/out:' + sharedLibrary]
        else:
            allObjectFiles = glob.glob(os.path.join(cacheConfig['objectCache'], '*.o'))
            linkCmd = [compilerConfig['command'], '-shared', '-o', sharedLibrary]

        linkCmd += allObjectFiles
        if len(allObjectFiles) > 0:
            runCompileStep(linkCmd)
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise e

atexit.register(compileObjectCacheToSharedLibrary)


def generateCode(ast, includes, restrictQualifier, functionPrefix, targetFile):
    with open(targetFile, 'w') as sourceFile:
        code = generateC(ast)
        includes = "\n".join(["#include <%s>" % (includeFile,) for includeFile in includes])
        print(includes, file=sourceFile)
        print("#define RESTRICT %s" % (restrictQualifier,), file=sourceFile)
        print("#define FUNC_PREFIX %s" % (functionPrefix,), file=sourceFile)
        print('extern "C" { ', file=sourceFile)
        print(code, file=sourceFile)
        print('}', file=sourceFile)


def runCompileStep(command):
    compilerConfig = getCompilerConfig()
    configEnv = compilerConfig['env'] if 'env' in compilerConfig else {}
    compileEnvironment = os.environ.copy()
    compileEnvironment.update(configEnv)

    try:
        shell = True if compilerConfig['os'].lower() == 'windows' else False
        subprocess.check_output(command, env=compileEnvironment, stderr=subprocess.STDOUT, shell=shell)
    except subprocess.CalledProcessError as e:
        print(" ".join(command))
        print(e.output)
        raise e


def compileLinux(ast, codeHashStr, srcFile, libFile):
    cacheConfig = getCacheConfig()
    compilerConfig = getCompilerConfig()

    objectFile = os.path.join(cacheConfig['objectCache'], codeHashStr + '.o')
    # Compilation
    if not os.path.exists(objectFile):
        generateCode(ast, ['iostream', 'cmath'], compilerConfig['restrictQualifier'], '', srcFile)
        compileCmd = [compilerConfig['command'], '-c'] + compilerConfig['flags'].split()
        compileCmd += ['-o', objectFile, srcFile]
        runCompileStep(compileCmd)

    # Linking
    runCompileStep([compilerConfig['command'], '-shared', objectFile, '-o', libFile] + compilerConfig['flags'].split())


def compileWindows(ast, codeHashStr, srcFile, libFile):
    cacheConfig = getCacheConfig()
    compilerConfig = getCompilerConfig()

    objectFile = os.path.join(cacheConfig['objectCache'], codeHashStr + '.obj')
    # Compilation
    if not os.path.exists(objectFile):
        generateCode(ast, ['iostream', 'cmath'], compilerConfig['restrictQualifier'],
                     '__declspec(dllexport)', srcFile)

        # /c compiles only, /EHsc turns of exception handling in c code
        compileCmd = ['cl.exe', '/c', '/EHsc'] + compilerConfig['flags'].split()
        compileCmd += [srcFile, '/Fo' + objectFile]
        runCompileStep(compileCmd)

    # Linking
    runCompileStep(['link.exe', '/DLL', '/out:' + libFile, objectFile])


def compileAndLoad(ast):
    cacheConfig = getCacheConfig()

    codeHashStr = hashlib.sha256(generateC(ast).encode()).hexdigest()
    ast.functionName = hashToFunctionName(codeHashStr)

    srcFile = os.path.join(cacheConfig['objectCache'], codeHashStr + ".cpp")

    if cacheConfig['readFromSharedLibrary']:
        return cdll.LoadLibrary(cacheConfig['sharedLibrary'])[ast.functionName]
    else:
        if getCompilerConfig()['os'].lower() == 'windows':
            libFile = os.path.join(cacheConfig['objectCache'], codeHashStr + ".dll")
            if not os.path.exists(libFile):
                compileWindows(ast, codeHashStr, srcFile, libFile)
        else:
            libFile = os.path.join(cacheConfig['objectCache'], codeHashStr + ".so")
            if not os.path.exists(libFile):
                compileLinux(ast, codeHashStr, srcFile, libFile)
        return cdll.LoadLibrary(libFile)[ast.functionName]


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
    func = compileAndLoad(kernelFunctionNode)
    func.restype = None
    parameters = kernelFunctionNode.parameters

    def wrapper(**kwargs):
        from copy import copy
        fullArguments = copy(argumentDict)
        fullArguments.update(kwargs)
        args = buildCTypeArgumentList(parameters, fullArguments)
        func(*args)
    return wrapper


