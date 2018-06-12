import os
import textwrap
import hashlib
import numpy as np
from sysconfig import get_paths
from pystencils import FieldType
from pystencils.cpu.cpujit import run_compile_step
from pystencils.data_types import get_base_type
from pystencils.backends.cbackend import generate_c, get_headers
from pystencils.utils import file_handle_for_atomic_write, atomic_file_write

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
    if( !kwargs || !PyDict_Check(kwargs) ) {{ PyErr_SetString(PyExc_TypeError, "No keyword arguments passed"); return NULL; }}
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
                pre_call_code += "Py_ssize_t {name}_shape[] = {{ {elements} }};\n".format(name=arg.field_name,
                                                                                          elements=shapes)

                item_size = get_base_type(arg.dtype).numpy_dtype.itemsize
                strides = ["buffer_{name}.strides[{i}] / {bytes}".format(i=i, name=arg.field_name, bytes=item_size)
                           for i in range(len(arg.field.strides))]
                strides = ", ".join(strides)
                pre_call_code += "Py_ssize_t {name}_strides[] = {{ {elements} }};\n".format(name=arg.field_name,
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
    code_hash_str = "mod_" + hashlib.sha256(generate_c(ast).encode()).hexdigest()
    code = ExtensionModuleCode(module_name=code_hash_str)
    code.add_function(ast, ast.function_name)
    src_file = os.path.join(cache_config['object_cache'], code_hash_str + ".cpp")
    lib_file = os.path.join(cache_config['object_cache'], code_hash_str + ".so")
    if not os.path.exists(lib_file):
        compiler_config = get_compiler_config()
        extra_flags = ['-I' + get_paths()['include']]
        object_file = os.path.join(cache_config['object_cache'], code_hash_str + '.o')
        if not os.path.exists(object_file):
            with file_handle_for_atomic_write(src_file) as f:
                code.write_to_file(compiler_config['restrict_qualifier'], '', f)
            with atomic_file_write(object_file) as file_name:
                compile_cmd = [compiler_config['command'], '-c'] + compiler_config['flags'].split()
                compile_cmd += [*extra_flags, '-o', file_name, src_file]
                run_compile_step(compile_cmd)

        # Linking
        with atomic_file_write(lib_file) as file_name:
            run_compile_step([compiler_config['command'], '-shared', object_file, '-o', file_name] +
                             compiler_config['flags'].split())

    result = load_kernel_from_file(code_hash_str, ast.function_name, lib_file)
    return KernelWrapper(result, ast.parameters, ast)


def make_python_function(kernel_function_node, argument_dict=None):
    import functools
    result = compile_and_load(kernel_function_node)
    if argument_dict:
        result = functools.partial(result, **argument_dict)
    return result
