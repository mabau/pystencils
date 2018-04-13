r"""

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
- **'restrict_qualifier'**: the restrict qualifier is not standardized accross compilers.
  For most Linux compilers the qualifier is ``__restrict__``


Compiler Config (Windows)
-------------------------

*pystencils* uses the mechanism of *setuptools.msvc* to search for a compilation environment.
Then 'cl.exe' is used to compile.

- **'os'**: should be detected automatically as 'windows'
- **'msvc_version'**:  either a version number, year number, 'auto' or 'latest' for automatic detection of latest
                      installed version or 'setuptools' for setuptools-based detection. Alternatively path to folder
                      where Visual Studio is installed. This path has to contain a file called 'vcvarsall.bat'
- **'arch'**: 'x86' or 'x64'
- **'flags'**: flags passed to 'cl.exe', make sure OpenMP is activated
- **'restrict_qualifier'**: the restrict qualifier is not standardized across compilers.
  For Windows compilers the qualifier should be ``__restrict``


Cache Config
------------

*pystencils* uses a directory to store intermediate files like the generated C++ files, compiled object files and
the shared libraries which are then loaded from Python using ctypes. The file names are SHA hashes of the
generated code. If the same kernel was already compiled, the existing object file is used - no recompilation is done.

If 'shared_library' is specified, all kernels that are currently in the cache are compiled into a single shared library.
This mechanism can be used to run *pystencils* on systems where compilation is not possible, e.g. on clusters where
compilation on the compute nodes is not possible.
First the script is run on a system where compilation is possible (e.g. the login node) with
'read_from_shared_library=False' and with 'shared_library' set a valid path.
All kernels generated during the run are put into the cache and at the end
compiled into the shared library. Then, the same script can be run from the compute nodes, with
'read_from_shared_library=True', such that kernels are taken from the library instead of compiling them.


- **'read_from_shared_library'**: if true kernels are not compiled but assumed to be in the shared library
- **'object_cache'**: path to a folder where intermediate files are stored
- **'clear_cache_on_start'**: when true the cache is cleared on each start of a *pystencils* script
- **'shared_library'**: path to a shared library file, which is created if `read_from_shared_library=false`
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
import numpy as np
from appdirs import user_config_dir, user_cache_dir
from ctypes import cdll
from pystencils.backends.cbackend import generate_c, get_headers
from collections import OrderedDict
from pystencils.transformations import symbol_name_to_variable_name
from pystencils.data_types import to_ctypes, get_base_type, StructType
from pystencils.field import FieldType
from pystencils.utils import recursive_dict_update


def make_python_function(kernel_function_node, argument_dict={}):
    """
    Creates C code from the abstract syntax tree, compiles it and makes it accessible as Python function

    The parameters of the kernel are:
        - numpy arrays for each field used in the kernel. The keyword argument name is the name of the field
        - all symbols which are not defined in the kernel itself are expected as parameters

    :param kernel_function_node: the abstract syntax tree
    :param argument_dict: parameters passed here are already fixed. Remaining parameters have to be passed to the
                        returned kernel functor.
    :return: kernel functor
    """
    # build up list of CType arguments
    func = compile_and_load(kernel_function_node)
    func.restype = None
    try:
        args = build_ctypes_argument_list(kernel_function_node.parameters, argument_dict)
    except KeyError:
        # not all parameters specified yet
        return make_python_function_incomplete_params(kernel_function_node, argument_dict, func)
    return lambda: func(*args)


def set_compiler_config(config):
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


def get_configuration_file_path():
    config_path_in_home = os.path.join(user_config_dir('pystencils'), 'config.json')

    # 1) Read path from environment variable if found
    if 'PYSTENCILS_CONFIG' in os.environ:
        return os.environ['PYSTENCILS_CONFIG'], True
    # 2) Look in current directory for pystencils.json
    elif os.path.exists("pystencils.json"):
        return "pystencils.json", True
    # 3) Try ~/.pystencils.json
    elif os.path.exists(config_path_in_home):
        return config_path_in_home, True
    else:
        return config_path_in_home, False


def create_folder(path, is_file):
    if is_file:
        path = os.path.split(path)[0]
    try:
        os.makedirs(path)
    except os.error:
        pass


def read_config():
    if platform.system().lower() == 'linux':
        default_compiler_config = OrderedDict([
            ('os', 'linux'),
            ('command', 'g++'),
            ('flags', '-Ofast -DNDEBUG -fPIC -march=native -fopenmp -std=c++11'),
            ('restrict_qualifier', '__restrict__')
        ])

    elif platform.system().lower() == 'windows':
        default_compiler_config = OrderedDict([
            ('os', 'windows'),
            ('msvc_version', 'latest'),
            ('arch', 'x64'),
            ('flags', '/Ox /fp:fast /openmp /arch:avx'),
            ('restrict_qualifier', '__restrict')
        ])
    default_cache_config = OrderedDict([
        ('read_from_shared_library', False),
        ('object_cache', os.path.join(user_cache_dir('pystencils'), 'objectcache')),
        ('clear_cache_on_start', False),
        ('shared_library', os.path.join(user_cache_dir('pystencils'), 'cache.so')),
    ])

    default_config = OrderedDict([('compiler', default_compiler_config),
                                  ('cache', default_cache_config)])

    config_path, config_exists = get_configuration_file_path()
    config = default_config.copy()
    if config_exists:
        with open(config_path, 'r') as json_config_file:
            loaded_config = json.load(json_config_file)
        config = recursive_dict_update(config, loaded_config)
    else:
        create_folder(config_path, True)
        json.dump(config, open(config_path, 'w'), indent=4)

    config['cache']['shared_library'] = os.path.expanduser(config['cache']['shared_library']).format(pid=os.getpid())
    config['cache']['object_cache'] = os.path.expanduser(config['cache']['object_cache']).format(pid=os.getpid())

    if config['cache']['clear_cache_on_start']:
        shutil.rmtree(config['cache']['object_cache'], ignore_errors=True)

    create_folder(config['cache']['object_cache'], False)
    create_folder(config['cache']['shared_library'], True)

    if 'env' not in config['compiler']:
        config['compiler']['env'] = {}

    if config['compiler']['os'] == 'windows':
        from pystencils.cpu.msvc_detection import get_environment
        msvc_env = get_environment(config['compiler']['msvc_version'], config['compiler']['arch'])
        config['compiler']['env'].update(msvc_env)

    return config


_config = read_config()


def get_compiler_config():
    return _config['compiler']


def get_cache_config():
    return _config['cache']


def hash_to_function_name(h):
    res = "func_%s" % (h,)
    return res.replace('-', 'm')


def compile_object_cache_to_shared_library():
    compiler_config = get_compiler_config()
    cache_config = get_cache_config()

    shared_library = cache_config['shared_library']
    if len(shared_library) == 0 or cache_config['read_from_shared_library']:
        return

    config_env = compiler_config['env'] if 'env' in compiler_config else {}
    compile_environment = os.environ.copy()
    compile_environment.update(config_env)

    try:
        if compiler_config['os'] == 'windows':
            all_object_files = glob.glob(os.path.join(cache_config['object_cache'], '*.obj'))
            link_cmd = ['link.exe', '/DLL', '/out:' + shared_library]
        else:
            all_object_files = glob.glob(os.path.join(cache_config['object_cache'], '*.o'))
            link_cmd = [compiler_config['command'], '-shared', '-o', shared_library]

        link_cmd += all_object_files
        if len(all_object_files) > 0:
            run_compile_step(link_cmd)
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise e


atexit.register(compile_object_cache_to_shared_library)


def generate_code(ast, restrict_qualifier, function_prefix, target_file):
    headers = get_headers(ast)
    headers.update(['<cmath>', '<cstdint>'])

    with open(target_file, 'w') as source_file:
        code = generate_c(ast)
        includes = "\n".join(["#include %s" % (include_file,) for include_file in headers])
        print(includes, file=source_file)
        print("#define RESTRICT %s" % (restrict_qualifier,), file=source_file)
        print("#define FUNC_PREFIX %s" % (function_prefix,), file=source_file)
        print('extern "C" { ', file=source_file)
        print(code, file=source_file)
        print('}', file=source_file)


def run_compile_step(command):
    compiler_config = get_compiler_config()
    config_env = compiler_config['env'] if 'env' in compiler_config else {}
    compile_environment = os.environ.copy()
    compile_environment.update(config_env)

    try:
        shell = True if compiler_config['os'].lower() == 'windows' else False
        subprocess.check_output(command, env=compile_environment, stderr=subprocess.STDOUT, shell=shell)
    except subprocess.CalledProcessError as e:
        print(" ".join(command))
        print(e.output.decode('utf8'))
        raise e


def compile_linux(ast, code_hash_str, src_file, lib_file):
    cache_config = get_cache_config()
    compiler_config = get_compiler_config()

    object_file = os.path.join(cache_config['object_cache'], code_hash_str + '.o')
    # Compilation
    if not os.path.exists(object_file):
        generate_code(ast, compiler_config['restrict_qualifier'], '', src_file)
        compile_cmd = [compiler_config['command'], '-c'] + compiler_config['flags'].split()
        compile_cmd += ['-o', object_file, src_file]
        run_compile_step(compile_cmd)

    # Linking
    run_compile_step([compiler_config['command'], '-shared', object_file, '-o', lib_file] +
                     compiler_config['flags'].split())


def compile_windows(ast, code_hash_str, src_file, lib_file):
    cache_config = get_cache_config()
    compiler_config = get_compiler_config()

    object_file = os.path.join(cache_config['object_cache'], code_hash_str + '.obj')
    # Compilation
    if not os.path.exists(object_file):
        generate_code(ast, compiler_config['restrict_qualifier'],
                      '__declspec(dllexport)', src_file)

        # /c compiles only, /EHsc turns of exception handling in c code
        compile_cmd = ['cl.exe', '/c', '/EHsc'] + compiler_config['flags'].split()
        compile_cmd += [src_file, '/Fo' + object_file]
        run_compile_step(compile_cmd)

    # Linking
    run_compile_step(['link.exe', '/DLL', '/out:' + lib_file, object_file])


def compile_and_load(ast):
    cache_config = get_cache_config()

    code_hash_str = hashlib.sha256(generate_c(ast).encode()).hexdigest()
    ast.function_name = hash_to_function_name(code_hash_str)

    src_file = os.path.join(cache_config['object_cache'], code_hash_str + ".cpp")

    if cache_config['read_from_shared_library']:
        return cdll.LoadLibrary(cache_config['shared_library'])[ast.function_name]
    else:
        if get_compiler_config()['os'].lower() == 'windows':
            lib_file = os.path.join(cache_config['object_cache'], code_hash_str + ".dll")
            if not os.path.exists(lib_file):
                compile_windows(ast, code_hash_str, src_file, lib_file)
        else:
            lib_file = os.path.join(cache_config['object_cache'], code_hash_str + ".so")
            if not os.path.exists(lib_file):
                compile_linux(ast, code_hash_str, src_file, lib_file)
        return cdll.LoadLibrary(lib_file)[ast.function_name]


def build_ctypes_argument_list(parameter_specification, argument_dict):
    argument_dict = {symbol_name_to_variable_name(k): v for k, v in argument_dict.items()}
    ct_arguments = []
    array_shapes = set()
    index_arr_shapes = set()

    for arg in parameter_specification:
        if arg.is_field_argument:
            try:
                field_arr = argument_dict[arg.field_name]
            except KeyError:
                raise KeyError("Missing field parameter for kernel call " + arg.field_name)

            symbolic_field = arg.field
            if arg.is_field_ptr_argument:
                ct_arguments.append(field_arr.ctypes.data_as(to_ctypes(arg.dtype)))
                if symbolic_field.has_fixed_shape:
                    symbolic_field_shape = tuple(int(i) for i in symbolic_field.shape)
                    if isinstance(symbolic_field.dtype, StructType):
                        symbolic_field_shape = symbolic_field_shape[:-1]
                    if symbolic_field_shape != field_arr.shape:
                        raise ValueError("Passed array '%s' has shape %s which does not match expected shape %s" %
                                         (arg.field_name, str(field_arr.shape), str(symbolic_field.shape)))
                if symbolic_field.has_fixed_shape:
                    symbolic_field_strides = tuple(int(i) * field_arr.itemsize for i in symbolic_field.strides)
                    if isinstance(symbolic_field.dtype, StructType):
                        symbolic_field_strides = symbolic_field_strides[:-1]
                    if symbolic_field_strides != field_arr.strides:
                        raise ValueError("Passed array '%s' has strides %s which does not match expected strides %s" %
                                         (arg.field_name, str(field_arr.strides), str(symbolic_field_strides)))

                if FieldType.is_indexed(symbolic_field):
                    index_arr_shapes.add(field_arr.shape[:symbolic_field.spatial_dimensions])
                elif not FieldType.is_buffer(symbolic_field):
                    array_shapes.add(field_arr.shape[:symbolic_field.spatial_dimensions])

            elif arg.is_field_shape_argument:
                data_type = to_ctypes(get_base_type(arg.dtype))
                ct_arguments.append(field_arr.ctypes.shape_as(data_type))
            elif arg.is_field_stride_argument:
                data_type = to_ctypes(get_base_type(arg.dtype))
                strides = field_arr.ctypes.strides_as(data_type)
                for i in range(len(field_arr.shape)):
                    assert strides[i] % field_arr.itemsize == 0
                    strides[i] //= field_arr.itemsize
                ct_arguments.append(strides)
            else:
                assert False
        else:
            try:
                param = argument_dict[arg.name]
            except KeyError:
                raise KeyError("Missing parameter for kernel call " + arg.name)
            expected_type = to_ctypes(arg.dtype)
            ct_arguments.append(expected_type(param))

    if len(array_shapes) > 1:
        raise ValueError("All passed arrays have to have the same size " + str(array_shapes))
    if len(index_arr_shapes) > 1:
        raise ValueError("All passed index arrays have to have the same size " + str(array_shapes))

    return ct_arguments


def make_python_function_incomplete_params(kernel_function_node, argument_dict, func):
    parameters = kernel_function_node.parameters

    cache = {}
    cache_values = []

    def wrapper(**kwargs):
        key = hash(tuple((k, v.ctypes.data, v.strides, v.shape) if isinstance(v, np.ndarray) else (k, id(v))
                         for k, v in kwargs.items()))
        try:
            args = cache[key]
            func(*args)
        except KeyError:
            full_arguments = argument_dict.copy()
            full_arguments.update(kwargs)
            args = build_ctypes_argument_list(parameters, full_arguments)
            cache[key] = args
            cache_values.append(kwargs)  # keep objects alive such that ids remain unique
            func(*args)
    wrapper.ast = kernel_function_node
    wrapper.parameters = kernel_function_node.parameters
    return wrapper
