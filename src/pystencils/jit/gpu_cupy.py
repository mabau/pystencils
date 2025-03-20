from typing import Any, Callable
from dataclasses import dataclass

try:
    import cupy as cp

    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False

from ..codegen import Target
from ..field import FieldType

from ..types import PsType
from .jit import JitBase, JitError, KernelWrapper
from ..codegen import (
    Kernel,
    GpuKernel,
    Parameter,
)
from ..codegen.gpu_indexing import GpuLaunchConfiguration
from ..codegen.properties import FieldShape, FieldStride, FieldBasePtr
from ..types import PsStructType, PsPointerType

from ..include import get_pystencils_include_path


@dataclass
class LaunchGrid:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]


class CupyKernelWrapper(KernelWrapper):
    def __init__(
        self,
        kfunc: GpuKernel,
        raw_kernel: Any,
    ):
        self._kfunc: GpuKernel = kfunc
        self._launch_config = kfunc.get_launch_configuration()
        self._raw_kernel = raw_kernel
        self._args_cache: dict[Any, tuple] = dict()

    @property
    def kernel_function(self) -> GpuKernel:
        return self._kfunc

    @property
    def launch_config(self) -> GpuLaunchConfiguration:
        return self._launch_config

    @property
    def raw_kernel(self):
        return self._raw_kernel

    def __call__(self, **kwargs: Any):
        kernel_args, launch_grid = self._get_cached_args(**kwargs)
        device = self._get_device(kernel_args)
        with cp.cuda.Device(device):
            self._raw_kernel(launch_grid.grid, launch_grid.block, kernel_args)

    def _get_device(self, kernel_args):
        devices = set(a.device.id for a in kernel_args if type(a) is cp.ndarray)
        if len(devices) != 1:
            raise JitError("Could not determine CUDA device to execute on")
        return devices.pop()

    def _get_cached_args(self, **kwargs):
        key = (self._launch_config.jit_cache_key(),) + tuple(
            (k, id(v)) for k, v in kwargs.items()
        )

        if key not in self._args_cache:
            args = self._get_args(**kwargs)
            self._args_cache[key] = args
            return args
        else:
            return self._args_cache[key]

    def _get_args(self, **kwargs) -> tuple[tuple, LaunchGrid]:
        kernel_args = []
        valuation: dict[str, Any] = dict()

        def add_arg(param: Parameter, arg: Any):
            nptype = param.dtype.numpy_dtype
            assert nptype is not None
            typecast = nptype.type
            arg = typecast(arg)
            valuation[param.name] = arg

        def add_kernel_arg(param: Parameter, arg: Any):
            add_arg(param, arg)
            kernel_args.append(arg)

        field_shapes = set()
        index_shapes = set()

        def check_shape(field_ptr: Parameter, arr: cp.ndarray):
            field = field_ptr.fields[0]

            if field.has_fixed_shape:
                expected_shape = tuple(int(s) for s in field.shape)
                if isinstance(field.dtype, PsStructType):
                    assert expected_shape[-1] == 1
                    expected_shape = expected_shape[:-1]

                actual_shape = arr.shape
                if expected_shape != actual_shape:
                    raise ValueError(
                        f"Array kernel argument {field.name} had unexpected shape:\n"
                        f"   Expected {expected_shape}, but got {actual_shape}"
                    )

                expected_strides = tuple(int(s) for s in field.strides)
                if isinstance(field.dtype, PsStructType):
                    assert expected_strides[-1] == 1
                    expected_strides = expected_strides[:-1]

                actual_strides = tuple(s // arr.dtype.itemsize for s in arr.strides)
                if expected_strides != actual_strides:
                    raise ValueError(
                        f"Array kernel argument {field.name} had unexpected strides:\n"
                        f"   Expected {expected_strides}, but got {actual_strides}"
                    )

            match field.field_type:
                case FieldType.GENERIC:
                    field_shapes.add(arr.shape[: field.spatial_dimensions])

                    if len(field_shapes) > 1:
                        raise ValueError(
                            "Incompatible array shapes:"
                            "All arrays passed for generic fields to a kernel must have the same shape."
                        )

                case FieldType.INDEXED:
                    index_shapes.add(arr.shape)

                    if len(index_shapes) > 1:
                        raise ValueError(
                            "Incompatible array shapes:"
                            "All arrays passed for index fields to a kernel must have the same shape."
                        )

        #   Collect parameter values

        def process_param(param: Parameter, adder: Callable[[Parameter, Any], None]):
            arr: cp.ndarray

            if param.is_field_parameter:
                #   Determine field-associated data to pass in
                for prop in param.properties:
                    match prop:
                        case FieldBasePtr(field):

                            elem_dtype: PsType

                            from .. import DynamicType

                            if isinstance(field.dtype, DynamicType):
                                assert isinstance(param.dtype, PsPointerType)
                                elem_dtype = param.dtype.base_type
                            else:
                                elem_dtype = field.dtype

                            arr = kwargs[field.name]
                            if arr.dtype != elem_dtype.numpy_dtype:
                                raise JitError(
                                    f"Data type mismatch at array argument {field.name}:"
                                    f"Expected {field.dtype}, got {arr.dtype}"
                                )
                            check_shape(param, arr)
                            kernel_args.append(arr)
                            break

                        case FieldShape(field, coord):
                            arr = kwargs[field.name]
                            adder(param, arr.shape[coord])
                            break

                        case FieldStride(field, coord):
                            arr = kwargs[field.name]
                            adder(
                                param,
                                arr.strides[coord] // arr.dtype.itemsize,
                            )
                            break
            else:
                #   scalar parameter
                val: Any = kwargs[param.name]
                adder(param, val)

        #   Process Arguments

        for kparam in self._kfunc.parameters:
            process_param(kparam, add_kernel_arg)

        for cparam in self._launch_config.parameters:
            if cparam.name not in valuation:
                process_param(cparam, add_arg)

        block_size, grid_size = self._launch_config.evaluate(**valuation)

        return tuple(kernel_args), LaunchGrid(grid_size, block_size)


class CupyJit(JitBase):

    def compile(self, kernel: Kernel) -> KernelWrapper:
        if not HAVE_CUPY:
            raise JitError(
                "`cupy` is not installed: just-in-time-compilation of CUDA kernels is unavailable."
            )

        if not isinstance(kernel, GpuKernel):
            raise JitError(
                "The CupyJit just-in-time compiler only accepts GPU kernels generated for CUDA or HIP"
            )

        if kernel.target == Target.CUDA and cp.cuda.runtime.is_hip:
            raise JitError(
                "Cannot compile a CUDA kernel on a HIP-based Cupy installation."
            )

        if kernel.target == Target.HIP and not cp.cuda.runtime.is_hip:
            raise JitError(
                "Cannot compile a HIP kernel on a CUDA-based Cupy installation."
            )

        options = self._compiler_options()
        prelude = self._prelude(kernel)
        kernel_code = self._kernel_code(kernel)
        code = prelude + kernel_code

        raw_kernel = cp.RawKernel(
            code, kernel.name, options=options, backend="nvrtc", jitify=True
        )
        return CupyKernelWrapper(kernel, raw_kernel)

    def _compiler_options(self) -> tuple[str, ...]:
        options = ["-w", "-std=c++11"]
        options.append("-I" + get_pystencils_include_path())
        return tuple(options)

    def _prelude(self, kfunc: GpuKernel) -> str:

        headers: set[str]
        if cp.cuda.runtime.is_hip:
            headers = set()
        else:
            headers = {"<cstdint>"}

        headers |= kfunc.required_headers

        if '"pystencils_runtime/half.h"' in headers:
            headers.remove('"pystencils_runtime/half.h"')
            if cp.cuda.runtime.is_hip:
                headers.add("<hip/hip_fp16.h>")
            else:
                headers.add("<cuda_fp16.h>")

        code = "\n".join(f"#include {header}" for header in headers)

        code += "\n\n#define RESTRICT __restrict__\n\n"

        return code

    def _kernel_code(self, kfunc: GpuKernel) -> str:
        kernel_code = kfunc.get_c_code()
        return f'extern "C" {kernel_code}'
