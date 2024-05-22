from __future__ import annotations

from typing import Any

from os import path
import hashlib
from itertools import chain
from textwrap import indent

import numpy as np

from ..exceptions import PsInternalCompilerError
from ..kernelfunction import (
    KernelFunction,
    KernelParameter,
    FieldParameter,
    FieldShapeParam,
    FieldStrideParam,
    FieldPointerParam,
)
from ..constraints import KernelParamsConstraint
from ...types import (
    PsType,
    PsUnsignedIntegerType,
    PsSignedIntegerType,
    PsIeeeFloatType,
)
from ...types.quick import Fp, SInt, UInt
from ...field import Field
from ..emission import emit_code


class PsKernelExtensioNModule:
    """Replacement for `pystencils.cpu.cpujit.ExtensionModuleCode`.
    Conforms to its interface for plug-in to `compile_and_load`.
    """

    def __init__(
        self, module_name: str = "generated", custom_backend: Any = None
    ) -> None:
        self._module_name = module_name

        if custom_backend is not None:
            raise PsInternalCompilerError(
                "The `custom_backend` parameter exists only for interface compatibility and cannot be set."
            )

        self._kernels: dict[str, KernelFunction] = dict()
        self._code_string: str | None = None
        self._code_hash: str | None = None

    @property
    def module_name(self) -> str:
        return self._module_name

    def add_function(self, kernel_function: KernelFunction, name: str | None = None):
        if name is None:
            name = kernel_function.name

        self._kernels[name] = kernel_function

    def create_code_string(self, restrict_qualifier: str, function_prefix: str):
        code = ""

        #   Collect headers
        headers = {"<stdint.h>"}
        for kernel in self._kernels.values():
            headers |= kernel.required_headers

        header_list = sorted(headers)
        header_list.insert(0, '"Python.h"')

        from pystencils.include import get_pystencils_include_path

        ps_incl_path = get_pystencils_include_path()

        ps_headers = []
        for header in header_list:
            header = header[1:-1]
            header_path = path.join(ps_incl_path, header)
            if path.exists(header_path):
                ps_headers.append(header_path)

        header_hash = b"".join(
            [hashlib.sha256(open(h, "rb").read()).digest() for h in ps_headers]
        )

        #   Prelude: Includes and definitions

        includes = "\n".join(f"#include {header}" for header in header_list)

        code += includes
        code += "\n"
        code += f"#define RESTRICT {restrict_qualifier}\n"
        code += f"#define FUNC_PREFIX {function_prefix}\n"
        code += "\n"

        #   Kernels and call wrappers

        for name, kernel in self._kernels.items():
            old_name = kernel.name
            kernel.name = f"kernel_{name}"

            code += emit_code(kernel)
            code += "\n"
            code += emit_call_wrapper(name, kernel)
            code += "\n"

            kernel.name = old_name

        self._code_hash = (
            "mod_" + hashlib.sha256(code.encode() + header_hash).hexdigest()
        )

        code += create_module_boilerplate_code(self._code_hash, self._kernels.keys())

        self._code_string = code

    def get_hash_of_code(self):
        assert self._code_string is not None, "The code must be generated first"
        return self._code_hash

    def write_to_file(self, file):
        assert self._code_string is not None, "The code must be generated first"
        print(self._code_string, file=file)


def emit_call_wrapper(function_name: str, kernel: KernelFunction) -> str:
    builder = CallWrapperBuilder()

    for p in kernel.parameters:
        builder.extract_parameter(p)

    for c in kernel.constraints:
        builder.check_constraint(c)

    builder.call(kernel, kernel.parameters)

    return builder.resolve(function_name)


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


def create_module_boilerplate_code(module_name, names):
    method_definition = (
        '{{"{name}", (PyCFunction){name}, METH_VARARGS | METH_KEYWORDS, ""}},'
    )
    method_definitions = "\n".join(
        [method_definition.format(name=name) for name in names]
    )
    return template_module_boilerplate.format(
        module_name=module_name, method_definitions=method_definitions
    )


class CallWrapperBuilder:
    TMPL_EXTRACT_SCALAR = """
PyObject * obj_{name} = PyDict_GetItemString(kwargs, "{name}");
if( obj_{name} == NULL) {{  PyErr_SetString(PyExc_TypeError, "Keyword argument '{name}' missing"); return NULL; }};
{target_type} {name} = ({target_type}) {extract_function}( obj_{name} );
if( PyErr_Occurred() ) {{ return NULL; }}
"""

    TMPL_EXTRACT_ARRAY = """
PyObject * obj_{name} = PyDict_GetItemString(kwargs, "{name}");
if( obj_{name} == NULL) {{  PyErr_SetString(PyExc_TypeError, "Keyword argument '{name}' missing"); return NULL; }};
Py_buffer buffer_{name};
int buffer_{name}_res = PyObject_GetBuffer(obj_{name}, &buffer_{name}, PyBUF_STRIDES | PyBUF_WRITABLE | PyBUF_FORMAT);
if (buffer_{name}_res == -1) {{ return NULL; }}
"""

    TMPL_CHECK_ARRAY_TYPE = """
if(!({cond})) {{ 
    PyErr_SetString(PyExc_TypeError, "Wrong {what} of array {name}. Expected {expected}"); 
    return NULL; 
}}
"""

    KWCHECK = """
if( !kwargs || !PyDict_Check(kwargs) ) {{ 
    PyErr_SetString(PyExc_TypeError, "No keyword arguments passed"); 
    return NULL; 
}}
"""

    def __init__(self) -> None:
        self._array_buffers: dict[Field, str] = dict()
        self._array_extractions: dict[Field, str] = dict()
        self._array_frees: dict[Field, str] = dict()

        self._array_assoc_var_extractions: dict[FieldParameter, str] = dict()
        self._scalar_extractions: dict[KernelParameter, str] = dict()

        self._constraint_checks: list[str] = []

        self._call: str | None = None

    def _scalar_extractor(self, dtype: PsType) -> str:
        match dtype:
            case Fp(32) | Fp(64):
                return "PyFloat_AsDouble"
            case SInt():
                return "PyLong_AsLong"
            case UInt():
                return "PyLong_AsUnsignedLong"

            case _:
                raise PsInternalCompilerError(
                    f"Don't know how to cast Python objects to {dtype}"
                )

    def _type_char(self, dtype: PsType) -> str | None:
        if isinstance(
            dtype, (PsUnsignedIntegerType, PsSignedIntegerType, PsIeeeFloatType)
        ):
            np_dtype = dtype.NUMPY_TYPES[dtype.width]
            return np.dtype(np_dtype).char
        else:
            return None

    def extract_field(self, field: Field) -> str:
        """Adds an array, and returns the name of the underlying Py_Buffer."""
        if field not in self._array_extractions:
            extraction_code = self.TMPL_EXTRACT_ARRAY.format(name=field.name)

            #   Check array type
            type_char = self._type_char(field.dtype)
            if type_char is not None:
                dtype_cond = f"buffer_{field.name}.format[0] == '{type_char}'"
                extraction_code += self.TMPL_CHECK_ARRAY_TYPE.format(
                    cond=dtype_cond,
                    what="data type",
                    name=field.name,
                    expected=str(field.dtype),
                )

            #   Check item size
            itemsize = field.dtype.itemsize
            item_size_cond = f"buffer_{field.name}.itemsize == {itemsize}"
            extraction_code += self.TMPL_CHECK_ARRAY_TYPE.format(
                cond=item_size_cond, what="itemsize", name=field.name, expected=itemsize
            )

            self._array_buffers[field] = f"buffer_{field.name}"
            self._array_extractions[field] = extraction_code

            release_code = f"PyBuffer_Release(&buffer_{field.name});"
            self._array_frees[field] = release_code

        return self._array_buffers[field]

    def extract_scalar(self, param: KernelParameter) -> str:
        if param not in self._scalar_extractions:
            extract_func = self._scalar_extractor(param.dtype)
            code = self.TMPL_EXTRACT_SCALAR.format(
                name=param.name,
                target_type=str(param.dtype),
                extract_function=extract_func,
            )
            self._scalar_extractions[param] = code

        return param.name

    def extract_array_assoc_var(self, param: FieldParameter) -> str:
        if param not in self._array_assoc_var_extractions:
            field = param.field
            buffer = self.extract_field(field)
            match param:
                case FieldPointerParam():
                    code = f"{param.dtype} {param.name} = ({param.dtype}) {buffer}.buf;"
                case FieldShapeParam():
                    coord = param.coordinate
                    code = f"{param.dtype} {param.name} = {buffer}.shape[{coord}];"
                case FieldStrideParam():
                    coord = param.coordinate
                    code = (
                        f"{param.dtype} {param.name} = "
                        f"{buffer}.strides[{coord}] / {field.dtype.itemsize};"
                    )
                case _:
                    assert False, "unreachable code"

            self._array_assoc_var_extractions[param] = code

        return param.name

    def extract_parameter(self, param: KernelParameter):
        if isinstance(param, FieldParameter):
            self.extract_array_assoc_var(param)
        else:
            self.extract_scalar(param)

    def check_constraint(self, constraint: KernelParamsConstraint):
        variables = constraint.get_parameters()

        for var in variables:
            self.extract_parameter(var)

        cond = constraint.to_code()

        code = f"""
if(!({cond}))
{{
    PyErr_SetString(PyExc_ValueError, "Violated constraint: {constraint}"); 
    return NULL;
}}
"""

        self._constraint_checks.append(code)

    def call(self, kernel: KernelFunction, params: tuple[KernelParameter, ...]):
        param_list = ", ".join(p.name for p in params)
        self._call = f"{kernel.name} ({param_list});"

    def resolve(self, function_name) -> str:
        assert self._call is not None

        body = "\n\n".join(
            chain(
                [self.KWCHECK],
                self._scalar_extractions.values(),
                self._array_extractions.values(),
                self._array_assoc_var_extractions.values(),
                self._constraint_checks,
                [self._call],
                self._array_frees.values(),
                ["Py_RETURN_NONE;"],
            )
        )

        code = f"static PyObject * {function_name}(PyObject * self, PyObject * args, PyObject * kwargs)\n"
        code += "{\n" + indent(body, prefix="    ") + "\n}\n"

        return code