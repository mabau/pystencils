"""

*pystencils* looks for a configuration file in JSON format at the following locations in the listed order.

1. at the path specified in the environment variable ``PYSTENCILS_CONFIG``
2. in the current working direction for a file named ``pystencils.json``
3. or in your home directory at ``~/.config/pystencils/config.json`` (Linux) or
   ``%HOMEPATH%\.pystencils\config.json`` (Windows)

If no configuration file is found, a default configuration is created at the above mentioned location in your home.
So run *pystencils* once, then edit the created configuration file.


Compiler Config (Linux)
-----------------------

- **'os'**: should be detected automatically as 'linux'
- **'command'**: path to C++ compiler (defaults to 'g++')
- **'flags'**: space separated list of compiler flags. Make sure to activate OpenMP in your compiler
- **'restrictQualifier'**: the restrict qualifier is not standardized accross compilers.
  For most Linux compilers the qualifier is ``__restrict__``


Compiler Config (Windows)
-------------------------

*pystencils* uses the mechanism of *setuptools.msvc* to search for a compilation environment.
Then 'cl.exe' is used to compile.

- **'os'**: should be detected automatically as 'windows'
- **'msvcVersion'**:  either a version number, year number, 'auto' or 'latest' for automatic detection of latest
                      installed version or 'setuptools' for setuptools-based detection. Alternatively path to folder
                      where Visual Studio is installed. This path has to contain a file called 'vcvarsall.bat'
- **'arch'**: 'x86' or 'x64'
- **'flags'**: flags passed to 'cl.exe', make sure OpenMP is activated
- **'restrictQualifier'**: the restrict qualifier is not standardized accross compilers.
  For Windows compilers the qualifier should be ``__restrict``


Cache Config
------------

*pystencils* uses a directory to store intermediate files like the generated C++ files, compiled object files and
the shared libraries which are then loaded from Python using ctypes. The file names are SHA hashes of the
generated code. If the same kernel was already compiled, the existing object file is used - no recompilation is done.

If 'sharedLibrary' is specified, all kernels that are currently in the cache are compiled into a single shared library.
This mechanism can be used to run *pystencils* on systems where compilation is not possible, e.g. on clusters where
compilation on the compute nodes is not possible.
First the script is run on a system where compilation is possible (e.g. the login node) with
'readFromSharedLibrary=False' and with 'sharedLibrary' set a valid path.
All kernels generated during the run are put into the cache and at the end
compiled into the shared library. Then, the same script can be run from the compute nodes, with
'readFromSharedLibrary=True', such that kernels are taken from the library instead of compiling them.


- **'readFromSharedLibrary'**: if true kernels are not compiled but assumed to be in the shared library
- **'objectCache'**: path to a folder where intermediate files are stored
- **'clearCacheOnStart'**: when true the cache is cleared on each start of a *pystencils* script
- **'sharedLibrary'**: path to a shared library file, which is created if `readFromSharedLibrary=false`

"""
from __future__ import print_function
import os
import subprocess
import hashlib
import json
import platform
import glob
import atexit
import shutil
from ctypes import cdll, sizeof
from pystencils.backends.cbackend import generateC
from collections import OrderedDict, Mapping
from pystencils.transformations import symbolNameToVariableName
from pystencils.types import toCtypes, getBaseType, createType, StructType


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
            ('flags', '-Ofast -DNDEBUG -fPIC -march=native -fopenmp -std=c++11'),
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
        generateCode(ast, ['iostream', 'cmath', 'cstdint'], compilerConfig['restrictQualifier'], '', srcFile)
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
        generateCode(ast, ['iostream', 'cmath', 'cstdint'], compilerConfig['restrictQualifier'],
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
    indexArrShapes = set()

    for arg in parameterSpecification:
        if arg.isFieldArgument:
            try:
                fieldArr = argumentDict[arg.fieldName]
            except KeyError:
                raise KeyError("Missing field parameter for kernel call " + arg.fieldName)

            symbolicField = arg.field
            if arg.isFieldPtrArgument:
                ctArguments.append(fieldArr.ctypes.data_as(toCtypes(arg.dtype)))
                if symbolicField.hasFixedShape:
                    symbolicFieldShape = tuple(int(i) for i in symbolicField.shape)
                    if isinstance(symbolicField.dtype, StructType):
                        symbolicFieldShape = symbolicFieldShape[:-1]
                    if symbolicFieldShape != fieldArr.shape:
                        raise ValueError("Passed array '%s' has shape %s which does not match expected shape %s" %
                                         (arg.fieldName, str(fieldArr.shape), str(symbolicField.shape)))
                if symbolicField.hasFixedShape:
                    symbolicFieldStrides = tuple(int(i) * fieldArr.itemsize for i in symbolicField.strides)
                    if isinstance(symbolicField.dtype, StructType):
                        symbolicFieldStrides = symbolicFieldStrides[:-1]
                    if symbolicFieldStrides != fieldArr.strides:
                        raise ValueError("Passed array '%s' has strides %s which does not match expected strides %s" %
                                         (arg.fieldName, str(fieldArr.strides), str(symbolicFieldStrides)))

                if symbolicField.isIndexField:
                    indexArrShapes.add(fieldArr.shape[:symbolicField.spatialDimensions])
                else:
                    arrayShapes.add(fieldArr.shape[:symbolicField.spatialDimensions])

            elif arg.isFieldShapeArgument:
                dataType = toCtypes(getBaseType(arg.dtype))
                ctArguments.append(fieldArr.ctypes.shape_as(dataType))
            elif arg.isFieldStrideArgument:
                dataType = toCtypes(getBaseType(arg.dtype))
                strides = fieldArr.ctypes.strides_as(dataType)
                for i in range(len(fieldArr.shape)):
                    assert strides[i] % fieldArr.itemsize == 0
                    strides[i] //= fieldArr.itemsize
                ctArguments.append(strides)
            else:
                assert False
        else:
            try:
                param = argumentDict[arg.name]
            except KeyError:
                raise KeyError("Missing parameter for kernel call " + arg.name)
            expectedType = toCtypes(arg.dtype)
            ctArguments.append(expectedType(param))

    if len(arrayShapes) > 1:
        raise ValueError("All passed arrays have to have the same size " + str(arrayShapes))
    if len(indexArrShapes) > 1:
        raise ValueError("All passed index arrays have to have the same size " + str(arrayShapes))

    return ctArguments


def makePythonFunctionIncompleteParams(kernelFunctionNode, argumentDict):
    func = compileAndLoad(kernelFunctionNode)
    func.restype = None
    parameters = kernelFunctionNode.parameters

    cache = {}

    def wrapper(**kwargs):
        key = hash(tuple((k, id(v)) for k, v in kwargs.items()))
        try:
            args = cache[key]
            func(*args)
        except KeyError:
            fullArguments = argumentDict.copy()
            fullArguments.update(kwargs)
            args = buildCTypeArgumentList(parameters, fullArguments)
            cache[key] = args
            func(*args)
    return wrapper


