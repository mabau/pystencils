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

If no configuration file is found, a default configuration is created at the above-mentioned location in your home.
So run *pystencils* once, then edit the created configuration file.


Compiler Config (Linux)
-----------------------

- **'os'**: should be detected automatically as 'linux'
- **'command'**: path to C++ compiler (defaults to 'g++')
- **'flags'**: space separated list of compiler flags. Make sure to activate OpenMP in your compiler
- **'restrict_qualifier'**: the 'restrict' qualifier is not standardized across compilers.
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
- **'restrict_qualifier'**: the 'restrict' qualifier is not standardized across compilers.
  For Windows compilers the qualifier should be ``__restrict``

"""
from appdirs import user_cache_dir, user_config_dir
from collections import OrderedDict
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sysconfig
import tempfile
import time
import warnings


from ..ast import PsKernelFunction
from .cpu_extension_module import PsKernelExtensioNModule

from .msvc_detection import get_environment
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
        if platform.machine().startswith('ppc64') or platform.machine() == 'arm64':
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
        if platform.machine() == 'ARM64':
            default_compiler_config['arch'] = 'ARM64'
            default_compiler_config['flags'] = default_compiler_config['flags'].replace(' /arch:avx', '')
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
        raise NotImplementedError('Generation of default compiler flags for %s is not implemented' %
                                  (platform.system(),))

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
        with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(cache_status_file), delete=False) as f:
            json.dump(config['compiler'], f, indent=4)
        os.replace(f.name, cache_status_file)

    if config['compiler']['os'] == 'windows':
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


def load_kernel_from_file(module_name, function_name, path):
    try:
        spec = importlib.util.spec_from_file_location(name=module_name, location=path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except ImportError:
        warnings.warn(f"Could not load {path}, trying on more time in 5 seconds ...")
        time.sleep(5)
        spec = importlib.util.spec_from_file_location(name=module_name, location=path)
        mod = importlib.util.module_from_spec(spec)
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


def compile_module(code, code_hash, base_dir, compile_flags=None):
    if compile_flags is None:
        compile_flags = []

    compiler_config = get_compiler_config()
    extra_flags = ['-I' + sysconfig.get_paths()['include'], '-I' + get_pystencils_include_path()] + compile_flags

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


def compile_and_load(ast: PsKernelFunction, custom_backend=None):
    cache_config = get_cache_config()

    compiler_config = get_compiler_config()
    function_prefix = '__declspec(dllexport)' if compiler_config['os'].lower() == 'windows' else ''

    code = PsKernelExtensioNModule()

    code.add_function(ast, ast.function_name)

    code.create_code_string(compiler_config['restrict_qualifier'], function_prefix)
    code_hash_str = code.get_hash_of_code()

    compile_flags = []
    if ast.instruction_set and 'compile_flags' in ast.instruction_set:
        compile_flags = ast.instruction_set['compile_flags']

    if cache_config['object_cache'] is False:
        with tempfile.TemporaryDirectory() as base_dir:
            lib_file = compile_module(code, code_hash_str, base_dir, compile_flags=compile_flags)
            result = load_kernel_from_file(code_hash_str, ast.function_name, lib_file)
    else:
        lib_file = compile_module(code, code_hash_str, base_dir=cache_config['object_cache'],
                                  compile_flags=compile_flags)
        result = load_kernel_from_file(code_hash_str, ast.function_name, lib_file)

    return KernelWrapper(result, ast.get_parameters(), ast)
