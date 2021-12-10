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
import hashlib
import json
import os
import platform
import shutil
import subprocess
import textwrap
from collections import OrderedDict
from sysconfig import get_paths
from tempfile import TemporaryDirectory, NamedTemporaryFile

import numpy as np
from appdirs import user_cache_dir, user_config_dir

from pystencils import FieldType
from pystencils.astnodes import LoopOverCoordinate
from pystencils.backends.cbackend import generate_c, get_headers, CFunction
from pystencils.typing import CastFunc, VectorType, VectorMemoryAccess
from pystencils.include import get_pystencils_include_path
from pystencils.kernel_wrapper import KernelWrapper
from pystencils.utils import atomic_file_write, recursive_dict_update


def make_python_function(kernel_function_node, custom_backend=None):
    """
    Creates C code from the abstract syntax tree, compiles it and makes it accessible as Python function

    The parameters of the kernel are:
        - numpy arrays for each field used in the kernel. The keyword argument name is the name of the field
        - all symbols which are not defined in the kernel itself are expected as parameters

    :param kernel_function_node: the abstract syntax tree
    :param custom_backend: use own custom printer for code generation
    :return: kernel functor
    """
    result = compile_and_load(kernel_function_node, custom_backend)
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
        if platform.machine().startswith('ppc64'):
            default_compiler_config['flags'] = default_compiler_config['flags'].replace('-march=native',
                                                                                        '-mcpu=native')
    elif platform.system().lower() == 'windows':
        default_compiler_config = OrderedDict([
            ('os', 'windows'),
            ('msvc_version', 'latest'),
            ('arch', 'x64'),
            ('flags', '/Ox /fp:fast /OpenMP /arch:avx'),
            ('restrict_qualifier', '__restrict')
        ])
    elif platform.system().lower() == 'darwin':
        default_compiler_config = OrderedDict([
            ('os', 'darwin'),
            ('command', 'clang++'),
            ('flags', '-Ofast -DNDEBUG -fPIC -march=native -Xclang -fopenmp -std=c++11'),
            ('restrict_qualifier', '__restrict__')
        ])
        if platform.machine() == 'arm64':
            default_compiler_config['flags'] = default_compiler_config['flags'].replace('-march=native ', '')
        for libomp in ['/opt/local/lib/libomp/libomp.dylib', '/usr/local/lib/libomp.dylib',
                       '/opt/homebrew/lib/libomp.dylib']:
            if os.path.exists(libomp):
                default_compiler_config['flags'] += ' ' + libomp
                break
    else:
        raise ValueError("The detection of the platform with platform.system() did not work. "
                         "Pystencils is only supported for linux, windows, and darwin platforms.")

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
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    if config['cache']['object_cache'] is not False:
        config['cache']['object_cache'] = os.path.expanduser(config['cache']['object_cache']).format(pid=os.getpid())

        clear_cache_on_start = False
        cache_status_file = os.path.join(config['cache']['object_cache'], 'last_config.json')
        if os.path.exists(cache_status_file):
            # check if compiler config has changed
            last_config = json.load(open(cache_status_file, 'r'))
            if set(last_config.items()) != set(config['compiler'].items()):
                clear_cache_on_start = True
            else:
                for key in last_config.keys():
                    if last_config[key] != config['compiler'][key]:
                        clear_cache_on_start = True

        if config['cache']['clear_cache_on_start'] or clear_cache_on_start:
            shutil.rmtree(config['cache']['object_cache'], ignore_errors=True)

        create_folder(config['cache']['object_cache'], False)
        with NamedTemporaryFile('w', dir=os.path.dirname(cache_status_file), delete=False) as f:
            json.dump(config['compiler'], f, indent=4)
        os.replace(f.name, cache_status_file)

    if config['compiler']['os'] == 'windows':
        from pystencils.cpu.msvc_detection import get_environment
        msvc_env = get_environment(config['compiler']['msvc_version'], config['compiler']['arch'])
        if 'env' not in config['compiler']:
            config['compiler']['env'] = {}
        config['compiler']['env'].update(msvc_env)

    return config


_config = read_config()


def get_compiler_config():
    return _config['compiler']


def get_cache_config():
    return _config['cache']


def add_or_change_compiler_flags(flags):
    if not isinstance(flags, list) and not isinstance(flags, tuple):
        flags = [flags]

    compiler_config = get_compiler_config()
    cache_config = get_cache_config()
    cache_config['object_cache'] = False  # disable cache

    for flag in flags:
        flag = flag.strip()
        if '=' in flag:
            base = flag.split('=')[0].strip()
        else:
            base = flag

        new_flags = [c for c in compiler_config['flags'].split() if not c.startswith(base)]
        new_flags.append(flag)
        compiler_config['flags'] = ' '.join(new_flags)


def clear_cache():
    cache_config = get_cache_config()
    if cache_config['object_cache'] is not False:
        shutil.rmtree(cache_config['object_cache'], ignore_errors=True)
        create_folder(cache_config['object_cache'], False)


# TODO don't hardcode C type. [1] of tuple output
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
int buffer_{name}_res = PyObject_GetBuffer(obj_{name}, &buffer_{name}, PyBUF_STRIDES | PyBUF_WRITABLE | PyBUF_FORMAT);
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
    cond = [f"(buffer_{field_to_test.name}.shape[{i}] == buffer_{ref_field.name}.shape[{i}])"
            for field_to_test in fields[1:]
            for i in range(fields[0].spatial_dimensions)]
    cond = " && ".join(cond)
    return template_size_check.format(cond=cond)


def create_function_boilerplate_code(parameter_info, name, ast_node, insert_checks=True):
    pre_call_code = ""
    parameters = []
    post_call_code = ""
    variable_sized_normal_fields = set()
    variable_sized_index_fields = set()

    for param in parameter_info:
        if param.is_field_pointer:
            field = param.fields[0]
            pre_call_code += template_extract_array.format(name=field.name)
            post_call_code += template_release_buffer.format(name=field.name)
            parameters.append(f"({str(field.dtype)} *)buffer_{field.name}.buf")

            if insert_checks:
                np_dtype = field.dtype.numpy_dtype
                item_size = np_dtype.itemsize

                aligned = False
                if ast_node.assignments:
                    aligned = any([a.lhs.args[2] for a in ast_node.assignments
                                   if hasattr(a, 'lhs') and isinstance(a.lhs, CastFunc)
                                   and hasattr(a.lhs, 'dtype') and isinstance(a.lhs.dtype, VectorType)])

                if ast_node.instruction_set and aligned:
                    byte_width = ast_node.instruction_set['width'] * item_size
                    if 'cachelineZero' in ast_node.instruction_set:
                        has_openmp, has_nontemporal = False, False
                        for loop in ast_node.atoms(LoopOverCoordinate):
                            has_openmp = has_openmp or any(['#pragma omp' in p for p in loop.prefix_lines])
                            has_nontemporal = has_nontemporal or any([a.args[0].field == field and a.args[3] for a in
                                                                      loop.atoms(VectorMemoryAccess)])
                        if has_openmp and has_nontemporal:
                            byte_width = ast_node.instruction_set['cachelineSize']
                    offset = max(max(ast_node.ghost_layers)) * item_size
                    offset_cond = f"(((uintptr_t) buffer_{field.name}.buf) + {offset}) % ({byte_width}) == 0"

                    message = str(offset) + ". This is probably due to a different number of ghost_layers chosen for " \
                                            "the arrays and the kernel creation. If the number of ghost layers for " \
                                            "the kernel creation is not specified it will choose a suitable value " \
                                            "automatically. This value might not " \
                                            "be compatible with the allocated arrays."
                    if type(byte_width) is not int:
                        message += " Note that when both OpenMP and non-temporal stores are enabled, alignment to the "\
                                   "cacheline size is required."
                    pre_call_code += template_check_array.format(cond=offset_cond, what="offset", name=field.name,
                                                                 expected=message)

                if (np_dtype.isbuiltin and FieldType.is_generic(field)
                        and not np.issubdtype(field.dtype.numpy_dtype, np.complexfloating)):
                    dtype_cond = f"buffer_{field.name}.format[0] == '{field.dtype.numpy_dtype.char}'"
                    pre_call_code += template_check_array.format(cond=dtype_cond, what="data type", name=field.name,
                                                                 expected=str(field.dtype.numpy_dtype))

                item_size_cond = f"buffer_{field.name}.itemsize == {item_size}"
                pre_call_code += template_check_array.format(cond=item_size_cond, what="itemsize", name=field.name,
                                                             expected=item_size)

                if field.has_fixed_shape:
                    shape_cond = [f"buffer_{field.name}.shape[{i}] == {s}"
                                  for i, s in enumerate(field.spatial_shape)]
                    shape_cond = " && ".join(shape_cond)
                    pre_call_code += template_check_array.format(cond=shape_cond, what="shape", name=field.name,
                                                                 expected=str(field.shape))

                    expected_strides = [e * item_size for e in field.spatial_strides]
                    stride_check_code = "(buffer_{name}.strides[{i}] == {s} || buffer_{name}.shape[{i}]<=1)"
                    strides_cond = " && ".join([stride_check_code.format(s=s, i=i, name=field.name)
                                                for i, s in enumerate(expected_strides)])
                    pre_call_code += template_check_array.format(cond=strides_cond, what="strides", name=field.name,
                                                                 expected=str(expected_strides))
                else:
                    if FieldType.is_generic(field):
                        variable_sized_normal_fields.add(field)
                    elif FieldType.is_indexed(field):
                        variable_sized_index_fields.add(field)
        elif param.is_field_stride:
            field = param.fields[0]
            item_size = field.dtype.numpy_dtype.itemsize
            parameters.append(f"buffer_{field.name}.strides[{param.symbol.coordinate}] / {item_size}")
        elif param.is_field_shape:
            parameters.append(f"buffer_{param.field_name}.shape[{param.symbol.coordinate}]")
        elif type(param.symbol) is CFunction:
            continue
        else:
            extract_function, target_type = type_mapping[param.symbol.dtype.numpy_dtype.type]
            pre_call_code += template_extract_scalar.format(extract_function=extract_function,
                                                            target_type=target_type,
                                                            name=param.symbol.name)

            parameters.append(param.symbol.name)

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
    try:
        spec = spec_from_file_location(name=module_name, location=path)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
    except ImportError:
        import time
        import warnings
        warnings.warn(f"Could not load {path}, trying on more time in 5 seconds ...")
        time.sleep(5)
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
    def __init__(self, module_name='generated', custom_backend=None):
        self.module_name = module_name

        self._ast_nodes = []
        self._function_names = []
        self._custom_backend = custom_backend
        self._code_string = str()
        self._code_hash = None

    def add_function(self, ast, name=None):
        self._ast_nodes.append(ast)
        self._function_names.append(name if name is not None else ast.function_name)

    def create_code_string(self, restrict_qualifier, function_prefix):
        self._code_string = str()

        headers = {'<math.h>', '<stdint.h>'}
        for ast in self._ast_nodes:
            headers.update(get_headers(ast))
        header_list = list(headers)
        header_list.sort()
        header_list.insert(0, '"Python.h"')
        ps_headers = [os.path.join(os.path.dirname(__file__), '..', 'include', h[1:-1]) for h in header_list
                      if os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'include', h[1:-1]))]
        header_hash = b''.join([hashlib.sha256(open(h, 'rb').read()).digest() for h in ps_headers])

        includes = "\n".join([f"#include {include_file}" for include_file in header_list])
        self._code_string += includes
        self._code_string += "\n"
        self._code_string += f"#define RESTRICT {restrict_qualifier} \n"
        self._code_string += f"#define FUNC_PREFIX {function_prefix}"
        self._code_string += "\n"

        for ast, name in zip(self._ast_nodes, self._function_names):
            old_name = ast.function_name
            ast.function_name = f"kernel_{name}"
            self._code_string += generate_c(ast, custom_backend=self._custom_backend)
            self._code_string += create_function_boilerplate_code(ast.get_parameters(), name, ast)
            ast.function_name = old_name

        self._code_hash = "mod_" + hashlib.sha256(self._code_string.encode() + header_hash).hexdigest()
        self._code_string += create_module_boilerplate_code(self._code_hash, self._function_names)

    def get_hash_of_code(self):
        assert self._code_string, "The code must be generated first"
        return self._code_hash

    def write_to_file(self, file):
        assert self._code_string, "The code must be generated first"
        print(self._code_string, file=file)


def compile_module(code, code_hash, base_dir, compile_flags=None):
    if compile_flags is None:
        compile_flags = []

    compiler_config = get_compiler_config()
    extra_flags = ['-I' + get_paths()['include'], '-I' + get_pystencils_include_path()] + compile_flags

    if compiler_config['os'].lower() == 'windows':
        lib_suffix = '.pyd'
        object_suffix = '.obj'
        windows = True
    else:
        lib_suffix = '.so'
        object_suffix = '.o'
        windows = False

    src_file = os.path.join(base_dir, code_hash + ".cpp")
    lib_file = os.path.join(base_dir, code_hash + lib_suffix)
    object_file = os.path.join(base_dir, code_hash + object_suffix)

    if not os.path.exists(object_file):
        try:
            with open(src_file, 'x') as f:
                code.write_to_file(f)
        except FileExistsError:
            pass

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
                                  f"python{config_vars['py_version_nodot']}.lib")
            run_compile_step(['link.exe', py_lib, '/DLL', '/out:' + lib_file, object_file])
        elif platform.system().lower() == 'darwin':
            with atomic_file_write(lib_file) as file_name:
                run_compile_step([compiler_config['command'], '-shared', object_file, '-o', file_name, '-undefined',
                                  'dynamic_lookup']
                                 + compiler_config['flags'].split())
        else:
            with atomic_file_write(lib_file) as file_name:
                run_compile_step([compiler_config['command'], '-shared', object_file, '-o', file_name]
                                 + compiler_config['flags'].split())
    return lib_file


def compile_and_load(ast, custom_backend=None):
    cache_config = get_cache_config()

    compiler_config = get_compiler_config()
    function_prefix = '__declspec(dllexport)' if compiler_config['os'].lower() == 'windows' else ''

    code = ExtensionModuleCode(custom_backend=custom_backend)
    code.add_function(ast, ast.function_name)

    code.create_code_string(compiler_config['restrict_qualifier'], function_prefix)
    code_hash_str = code.get_hash_of_code()

    compile_flags = []
    if ast.instruction_set and 'compile_flags' in ast.instruction_set:
        compile_flags = ast.instruction_set['compile_flags']

    if cache_config['object_cache'] is False:
        with TemporaryDirectory() as base_dir:
            lib_file = compile_module(code, code_hash_str, base_dir, compile_flags=compile_flags)
            result = load_kernel_from_file(code_hash_str, ast.function_name, lib_file)
    else:
        lib_file = compile_module(code, code_hash_str, base_dir=cache_config['object_cache'],
                                  compile_flags=compile_flags)
        result = load_kernel_from_file(code_hash_str, ast.function_name, lib_file)

    return KernelWrapper(result, ast.get_parameters(), ast)
