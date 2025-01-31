from __future__ import annotations

from types import ModuleType
from typing import Sequence, cast
from pathlib import Path
from textwrap import indent

from pystencils.jit.jit import KernelWrapper

from ...types import PsPointerType, PsType
from ...field import Field
from ...sympyextensions import DynamicType
from ...codegen import Kernel, Parameter
from ...codegen.properties import FieldBasePtr, FieldShape, FieldStride

from .compiler_info import CompilerInfo
from .cpujit import ExtensionModuleBuilderBase


_module_template = Path(__file__).parent / "pybind11_kernel_module.tmpl.cpp"


class Pybind11KernelModuleBuilder(ExtensionModuleBuilderBase):
    @staticmethod
    def include_dirs() -> list[str]:
        import pybind11 as pb11

        pybind11_include = pb11.get_include()
        return [pybind11_include]

    def __init__(
        self,
        compiler_info: CompilerInfo,
    ):
        self._compiler_info = compiler_info

        self._actual_field_types: dict[Field, PsType]
        self._param_binds: list[str]
        self._public_params: list[str]
        self._param_check_lines: list[str]
        self._extraction_lines: list[str]

    def render_module(self, kernel: Kernel, module_name: str) -> str:
        self._actual_field_types = dict()
        self._param_binds = []
        self._public_params = []
        self._param_check_lines = []
        self._extraction_lines = []

        self._handle_params(kernel.parameters)

        kernel_def = self._get_kernel_definition(kernel)
        kernel_args = [param.name for param in kernel.parameters]
        includes = [f"#include {h}" for h in sorted(kernel.required_headers)]

        from string import Template

        templ = Template(_module_template.read_text())
        code_str = templ.substitute(
            includes="\n".join(includes),
            restrict_qualifier=self._compiler_info.restrict_qualifier(),
            module_name=module_name,
            kernel_name=kernel.name,
            param_binds=", ".join(self._param_binds),
            public_params=", ".join(self._public_params),
            param_check_lines=indent("\n".join(self._param_check_lines), prefix="    "),
            extraction_lines=indent("\n".join(self._extraction_lines), prefix="    "),
            kernel_args=", ".join(kernel_args),
            kernel_definition=kernel_def,
        )
        return code_str
    
    def get_wrapper(self, kernel: Kernel, extension_module: ModuleType) -> KernelWrapper:
        return Pybind11KernelWrapper(kernel, extension_module)

    def _get_kernel_definition(self, kernel: Kernel) -> str:
        from ...backend.emission import CAstPrinter

        printer = CAstPrinter()

        return printer(kernel)

    def _add_field_param(self, ptr_param: Parameter):
        field: Field = ptr_param.fields[0]

        ptr_type = ptr_param.dtype
        assert isinstance(ptr_type, PsPointerType)

        if isinstance(field.dtype, DynamicType):
            elem_type = ptr_type.base_type
        else:
            elem_type = field.dtype

        self._actual_field_types[field] = elem_type

        param_bind = f'py::arg("{field.name}").noconvert()'
        self._param_binds.append(param_bind)

        kernel_param = f"py::array_t< {elem_type.c_string()} > & {field.name}"
        self._public_params.append(kernel_param)

        expect_shape = "(" + ", ".join((str(s) if isinstance(s, int) else "*") for s in field.shape) + ")"
        for coord, size in enumerate(field.shape):
            if isinstance(size, int):
                self._param_check_lines.append(
                    f"checkFieldShape(\"{field.name}\", \"{expect_shape}\", {field.name}, {coord}, {size});"
                )

        expect_strides = "(" + ", ".join((str(s) if isinstance(s, int) else "*") for s in field.strides) + ")"
        for coord, stride in enumerate(field.strides):
            if isinstance(stride, int):
                self._param_check_lines.append(
                    f"checkFieldStride(\"{field.name}\", \"{expect_strides}\", {field.name}, {coord}, {stride});"
                )

    def _add_scalar_param(self, sc_param: Parameter):
        param_bind = f'py::arg("{sc_param.name}")'
        self._param_binds.append(param_bind)

        kernel_param = f"{sc_param.dtype.c_string()} {sc_param.name}"
        self._public_params.append(kernel_param)

    def _extract_base_ptr(self, ptr_param: Parameter, ptr_prop: FieldBasePtr):
        field_name = ptr_prop.field.name
        assert isinstance(ptr_param.dtype, PsPointerType)
        data_method = "data()" if ptr_param.dtype.base_type.const else "mutable_data()"
        extraction = f"{ptr_param.dtype.c_string()} {ptr_param.name} {{ {field_name}.{data_method} }};"
        self._extraction_lines.append(extraction)

    def _extract_shape(self, shape_param: Parameter, shape_prop: FieldShape):
        field_name = shape_prop.field.name
        coord = shape_prop.coordinate
        extraction = f"{shape_param.dtype.c_string()} {shape_param.name} {{ {field_name}.shape({coord}) }};"
        self._extraction_lines.append(extraction)

    def _extract_stride(self, stride_param: Parameter, stride_prop: FieldStride):
        field = stride_prop.field
        field_name = field.name
        coord = stride_prop.coordinate
        field_type = self._actual_field_types[field]
        assert field_type.itemsize is not None
        extraction = (
            f"{stride_param.dtype.c_string()} {stride_param.name} "
            f"{{ {field_name}.strides({coord}) / {field_type.itemsize} }};"
        )
        self._extraction_lines.append(extraction)

    def _handle_params(self, parameters: Sequence[Parameter]):
        for param in parameters:
            if param.get_properties(FieldBasePtr):
                self._add_field_param(param)

        for param in parameters:
            if ptr_props := param.get_properties(FieldBasePtr):
                self._extract_base_ptr(param, cast(FieldBasePtr, ptr_props.pop()))
            elif shape_props := param.get_properties(FieldShape):
                self._extract_shape(param, cast(FieldShape, shape_props.pop()))
            elif stride_props := param.get_properties(FieldStride):
                self._extract_stride(param, cast(FieldStride, stride_props.pop()))
            else:
                self._add_scalar_param(param)


class Pybind11KernelWrapper(KernelWrapper):
    def __init__(self, kernel: Kernel, jit_module: ModuleType):
        super().__init__(kernel)
        self._module = jit_module
        self._check_params = getattr(jit_module, "check_params")
        self._invoke = getattr(jit_module, "invoke")

    def __call__(self, **kwargs) -> None:
        self._check_params(**kwargs)
        return self._invoke(**kwargs)
