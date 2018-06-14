r"""

*pystencils* automatically searches for a compiler, so in most cases no explicit configuration is required.
On Linux make sure that 'gcc' and 'g++' are installed and in your path.
On Windows a recent Visual Studio installation is required.
In case anything does not work as expected or a special compiler should be used, changes can be specified
in a configuration file.

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

"""
import os
import hashlib
import json
import platform
import shutil
import textwrap
import numpy as np
import functools
import subprocess
from appdirs import user_config_dir, user_cache_dir
from collections import OrderedDict
from pystencils.utils import recursive_dict_update
from sysconfig import get_paths
from pystencils import FieldType, Field
from pystencils.data_types import get_base_type
from pystencils.backends.cbackend import generate_c, get_headers
from pystencils.utils import file_handle_for_atomic_write, atomic_file_write


def make_python_function(kernel_function_node):
    """
    Creates C code from the abstract syntax tree, compiles it and makes it accessible as Python function

    The parameters of the kernel are:
        - numpy arrays for each field used in the kernel. The keyword argument name is the name of the field
        - all symbols which are not defined in the kernel itself are expected as parameters

    :param kernel_function_node: the abstract syntax tree
    :return: kernel functor
    """
    result = compile_and_load(kernel_function_node)
    return result


def set_config(config):
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
        ('object_cache', os.path.join(user_cache_dir('pystencils'), 'objectcache')),
        ('clear_cache_on_start', False),
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

    config['cache']['object_cache'] = os.path.expanduser(config['cache']['object_cache']).format(pid=os.getpid())

    if config['cache']['clear_cache_on_start']:
        clear_cache()

    create_folder(config['cache']['object_cache'], False)

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


def clear_cache():
    cache_config = get_cache_config()
    shutil.rmtree(cache_config['object_cache'], ignore_errors=True)
    create_folder(cache_config['object_cache'], False)


type_mapping = {
    np.float32: ('PyFloat_AsDouble', 'float'),
    np.float64: ('PyFloat_AsDouble', 'double'),
    np.int16: ('PyLong_AsLong', 'int16_t'),
    np.int32: ('PyLong_AsLong', 'int32_t'),
    np.int64: ('PyLong_AsLong', 'int64_t'),
    np.uint16: ('PyLong_AsUnsignedLong', 'uint16_t'),
    np.uint32: ('PyLong_AsUnsignedLong', 'uint32_t'),
    np.uint64: ('PyLong_AsUnsignedLong', 'uint64_t'),
}


template_extract_scalar = """
PyObject * obj_{name} = PyDict_GetItemString(kwargs, "{name}");
if( obj_{name} == NULL) {{  PyErr_SetString(PyExc_TypeError, "Keyword argument '{name}' missing"); return NULL; }};
{target_type} {name} = ({target_type}) {extract_function}( obj_{name} );
if( PyErr_Occurred() ) {{ return NULL; }}
"""

template_extract_array = """
PyObject * obj_{name} = PyDict_GetItemString(kwargs, "{name}");
if( obj_{name} == NULL) {{  PyErr_SetString(PyExc_TypeError, "Keyword argument '{name}' missing"); return NULL; }};
Py_buffer buffer_{name};
int buffer_{name}_res = PyObject_GetBuffer(obj_{name}, &buffer_{name}, PyBUF_STRIDES | PyBUF_WRITABLE);
if (buffer_{name}_res == -1) {{ return NULL; }}
"""

template_release_buffer = """
PyBuffer_Release(&buffer_{name});
"""

template_function_boilerplate = """
static PyObject * {func_name}(PyObject * self, PyObject * args, PyObject * kwargs)
{{
    if( !kwargs || !PyDict_Check(kwargs) ) {{ 
        PyErr_SetString(PyExc_TypeError, "No keyword arguments passed"); 
        return NULL; 
    }}
    {pre_call_code}
    kernel_{func_name}({parameters});
    {post_call_code}
    Py_RETURN_NONE;
}}
"""

template_check_array = """
if(!({cond})) {{ 
    PyErr_SetString(PyExc_ValueError, "Wrong {what} of array {name}. Expected {expected}"); 
    return NULL; 
}}
"""

template_size_check = """
if(!({cond})) {{ 
    PyErr_SetString(PyExc_TypeError, "Arrays must have same shape"); return NULL; 
}}"""

template_module_boilerplate = """
static PyMethodDef method_definitions[] = {{
    {method_definitions}
    {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef module_definition = {{
    PyModuleDef_HEAD_INIT,
    "{module_name}",   /* name of module */
    NULL,     /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    method_definitions
}};

PyMODINIT_FUNC
PyInit_{module_name}(void)
{{
    return PyModule_Create(&module_definition);
}}
"""


def equal_size_check(fields):
    fields = list(fields)
    if len(fields) <= 1:
        return ""

    ref_field = fields[0]
    cond = ["({field.name}_shape[{i}] == {ref_field.name}_shape[{i}])".format(ref_field=ref_field,
                                                                              field=field_to_test, i=i)
            for field_to_test in fields[1:]
            for i in range(fields[0].spatial_dimensions)]
    cond = " && ".join(cond)
    return template_size_check.format(cond=cond)


def create_function_boilerplate_code(parameter_info, name, insert_checks=True):
    pre_call_code = ""
    parameters = []
    post_call_code = ""
    variable_sized_normal_fields = set()
    variable_sized_index_fields = set()

    for arg in parameter_info:
        if arg.is_field_argument:
            if arg.is_field_ptr_argument:
                pre_call_code += template_extract_array.format(name=arg.field_name)
                post_call_code += template_release_buffer.format(name=arg.field_name)
                parameters.append("({dtype} *)buffer_{name}.buf".format(dtype=str(arg.field.dtype),
                                                                        name=arg.field_name))

                shapes = ", ".join(["buffer_{name}.shape[{i}]".format(name=arg.field_name, i=i)
                                    for i in range(len(arg.field.strides))])

                shape_type = get_base_type(Field.SHAPE_DTYPE)
                pre_call_code += "{type} {name}_shape[] = {{ {elements} }};\n".format(type=shape_type,
                                                                                      name=arg.field_name,
                                                                                      elements=shapes)

                item_size = get_base_type(arg.dtype).numpy_dtype.itemsize
                strides = ["buffer_{name}.strides[{i}] / {bytes}".format(i=i, name=arg.field_name, bytes=item_size)
                           for i in range(len(arg.field.strides))]
                strides = ", ".join(strides)
                strides_type = get_base_type(Field.STRIDE_DTYPE)
                pre_call_code += "{type} {name}_strides[] = {{ {elements} }};\n".format(type=strides_type,
                                                                                        name=arg.field_name,
                                                                                        elements=strides)

                if insert_checks and arg.field.has_fixed_shape:
                    shape_cond = ["{name}_shape[{i}] == {s}".format(s=s, name=arg.field_name, i=i)
                                  for i, s in enumerate(arg.field.spatial_shape)]
                    shape_cond = " && ".join(shape_cond)
                    pre_call_code += template_check_array.format(cond=shape_cond, what="shape", name=arg.field.name,
                                                                 expected=str(arg.field.shape))

                    strides_cond = ["({name}_strides[{i}] == {s} || {name}_shape[{i}]<=1)".format(s=s, i=i,
                                                                                                  name=arg.field_name)
                                    for i, s in enumerate(arg.field.spatial_strides)]
                    strides_cond = " && ".join(strides_cond)
                    expected_strides_str = str([e * item_size for e in arg.field.strides])
                    pre_call_code += template_check_array.format(cond=strides_cond, what="strides", name=arg.field.name,
                                                                 expected=expected_strides_str)
                if insert_checks and not arg.field.has_fixed_shape:
                    if FieldType.is_generic(arg.field):
                        variable_sized_normal_fields.add(arg.field)
                    elif FieldType.is_indexed(arg.field):
                        variable_sized_index_fields.add(arg.field)

            elif arg.is_field_shape_argument:
                parameters.append("{name}_shape".format(name=arg.field_name))
            elif arg.is_field_stride_argument:
                parameters.append("{name}_strides".format(name=arg.field_name))
        else:
            extract_function, target_type = type_mapping[arg.dtype.numpy_dtype.type]
            pre_call_code += template_extract_scalar.format(extract_function=extract_function, target_type=target_type,
                                                            name=arg.name)
            parameters.append(arg.name)

    pre_call_code += equal_size_check(variable_sized_normal_fields)
    pre_call_code += equal_size_check(variable_sized_index_fields)

    pre_call_code = textwrap.indent(pre_call_code, '    ')
    post_call_code = textwrap.indent(post_call_code, '    ')
    return template_function_boilerplate.format(func_name=name, pre_call_code=pre_call_code,
                                                post_call_code=post_call_code, parameters=", ".join(parameters))


def create_module_boilerplate_code(module_name, names):
    method_definition = '{{"{name}", (PyCFunction){name}, METH_VARARGS | METH_KEYWORDS, ""}},'
    method_definitions = "\n".join([method_definition.format(name=name) for name in names])
    return template_module_boilerplate.format(module_name=module_name, method_definitions=method_definitions)


def load_kernel_from_file(module_name, function_name, path):
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(name=module_name, location=path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, function_name)


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


class ExtensionModuleCode:
    def __init__(self, module_name='generated'):
        self.module_name = module_name

        self._ast_nodes = []
        self._function_names = []

    def add_function(self, ast, name=None):
        self._ast_nodes.append(ast)
        self._function_names.append(name if name is not None else ast.function_name)

    def write_to_file(self, restrict_qualifier, function_prefix, file):
        headers = {'<math.h>', '<stdint.h>', '"Python.h"'}
        for ast in self._ast_nodes:
            headers.update(get_headers(ast))

        includes = "\n".join(["#include %s" % (include_file,) for include_file in headers])
        print(includes, file=file)
        print("\n", file=file)
        print("#define RESTRICT %s" % (restrict_qualifier,), file=file)
        print("#define FUNC_PREFIX %s" % (function_prefix,), file=file)
        print("\n", file=file)

        for ast, name in zip(self._ast_nodes, self._function_names):
            old_name = ast.function_name
            ast.function_name = "kernel_" + name
            print(generate_c(ast), file=file)
            print(create_function_boilerplate_code(ast.parameters, name), file=file)
            ast.function_name = old_name
        print(create_module_boilerplate_code(self.module_name, self._function_names), file=file)


class KernelWrapper:
    def __init__(self, kernel, parameters, ast_node):
        self.kernel = kernel
        self.parameters = parameters
        self.ast = ast_node

    def __call__(self, **kwargs):
        return self.kernel(**kwargs)


def compile_and_load(ast):
    from pystencils.cpu.cpujit import get_cache_config, get_compiler_config
    cache_config = get_cache_config()
    compiler_config = get_compiler_config()

    if compiler_config['os'].lower() == 'windows':
        function_prefix = '__declspec(dllexport)'
        lib_suffix = '.pyd'
        object_suffix = '.obj'
        windows = True
    else:
        function_prefix = ''
        lib_suffix = '.so'
        object_suffix = '.o'
        windows = False

    code_hash_str = "mod_" + hashlib.sha256(generate_c(ast).encode()).hexdigest()
    code = ExtensionModuleCode(module_name=code_hash_str)
    code.add_function(ast, ast.function_name)
    src_file = os.path.join(cache_config['object_cache'], code_hash_str + ".cpp")
    lib_file = os.path.join(cache_config['object_cache'], code_hash_str + lib_suffix)
    if not os.path.exists(lib_file):
        extra_flags = ['-I' + get_paths()['include']]
        object_file = os.path.join(cache_config['object_cache'], code_hash_str + object_suffix)
        if not os.path.exists(object_file):
            with file_handle_for_atomic_write(src_file) as f:
                code.write_to_file(compiler_config['restrict_qualifier'], function_prefix, f)

            if windows:
                compile_cmd = ['cl.exe', '/c', '/EHsc'] + compiler_config['flags'].split()
                compile_cmd += [*extra_flags, src_file, '/Fo' + object_file]
                run_compile_step(compile_cmd)
            else:
                with atomic_file_write(object_file) as file_name:
                    compile_cmd = [compiler_config['command'], '-c'] + compiler_config['flags'].split()
                    compile_cmd += [*extra_flags, '-o', file_name, src_file]
                    run_compile_step(compile_cmd)

            # Linking
            if windows:
                import sysconfig
                config_vars = sysconfig.get_config_vars()
                py_lib = os.path.join(config_vars["installed_base"], "libs",
                                      "python{}.lib".format(config_vars["py_version_nodot"]))
                run_compile_step(['link.exe', py_lib, '/DLL', '/out:' + lib_file, object_file])
            else:
                with atomic_file_write(lib_file) as file_name:
                    run_compile_step([compiler_config['command'], '-shared', object_file, '-o', file_name] +
                                     compiler_config['flags'].split())

    result = load_kernel_from_file(code_hash_str, ast.function_name, lib_file)
    return KernelWrapper(result, ast.parameters, ast)
