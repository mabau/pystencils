from typing import Callable, Any
from dataclasses import dataclass

try:
    import cupy as cp

    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False

from ...enums import Target
from ...field import FieldType

from ...types import PsType
from .jit import JitBase, JitError, KernelWrapper
from ..kernelfunction import (
    KernelFunction,
    GpuKernelFunction,
    FieldPointerParam,
    FieldShapeParam,
    FieldStrideParam,
    KernelParameter,
)
from ..emission import emit_code

from ...include import get_pystencils_include_path


@dataclass
class LaunchGrid:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]


class CupyKernelWrapper(KernelWrapper):
    def __init__(
        self,
        kfunc: GpuKernelFunction,
        raw_kernel: Any,
        block_size: tuple[int, int, int],
    ):
        self._kfunc: GpuKernelFunction = kfunc
        self._raw_kernel = raw_kernel
        self._block_size = block_size
        self._args_cache: dict[Any, tuple] = dict()

    @property
    def kernel_function(self) -> GpuKernelFunction:
        return self._kfunc

    @property
    def raw_kernel(self):
        return self._raw_kernel

    @property
    def block_size(self) -> tuple[int, int, int]:
        return self._block_size

    @block_size.setter
    def block_size(self, bs: tuple[int, int, int]):
        self._block_size = bs

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
        key = tuple(
            (
                (k, v.dtype, v.strides, v.shape)
                if isinstance(v, cp.ndarray)
                else (k, id(v))
            )
            for k, v in kwargs.items()
        )

        if key not in self._args_cache:
            args = self._get_args(**kwargs)
            self._args_cache[key] = args
            return args
        else:
            return self._args_cache[key]

    def _get_args(self, **kwargs) -> tuple[tuple, LaunchGrid]:
        args = []
        valuation: dict[str, Any] = dict()

        def add_arg(name: str, arg: Any, dtype: PsType):
            nptype = dtype.numpy_dtype
            assert nptype is not None
            typecast = nptype.type
            arg = typecast(arg)
            args.append(arg)
            valuation[name] = arg

        field_shapes = set()
        index_shapes = set()

        def check_shape(field_ptr: FieldPointerParam, arr: cp.ndarray):
            field = field_ptr.field

            if field.has_fixed_shape:
                expected_shape = tuple(int(s) for s in field.shape)
                actual_shape = arr.shape
                if expected_shape != actual_shape:
                    raise ValueError(
                        f"Array kernel argument {field.name} had unexpected shape:\n"
                        f"   Expected {expected_shape}, but got {actual_shape}"
                    )

                expected_strides = tuple(int(s) for s in field.strides)
                actual_strides = tuple(s // arr.dtype.itemsize for s in arr.strides)
                if expected_strides != actual_strides:
                    raise ValueError(
                        f"Array kernel argument {field.name} had unexpected strides:\n"
                        f"   Expected {expected_strides}, but got {actual_strides}"
                    )

            match field.field_type:
                case FieldType.GENERIC:
                    field_shapes.add(arr.shape)

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
        arr: cp.ndarray

        for kparam in self._kfunc.parameters:
            match kparam:
                case FieldPointerParam(_, dtype, field):
                    arr = kwargs[field.name]
                    if arr.dtype != field.dtype.numpy_dtype:
                        raise JitError(
                            f"Data type mismatch at array argument {field.name}:"
                            f"Expected {field.dtype}, got {arr.dtype}"
                        )
                    check_shape(kparam, arr)
                    args.append(arr)

                case FieldShapeParam(name, dtype, field, coord):
                    arr = kwargs[field.name]
                    add_arg(name, arr.shape[coord], dtype)

                case FieldStrideParam(name, dtype, field, coord):
                    arr = kwargs[field.name]
                    add_arg(name, arr.strides[coord] // arr.dtype.itemsize, dtype)

                case KernelParameter(name, dtype):
                    val: Any = kwargs[name]
                    add_arg(name, val, dtype)

        #   Determine launch grid
        from ..ast.expressions import evaluate_expression

        symbolic_threads_range = self._kfunc.threads_range

        threads_range: list[int] = [
            evaluate_expression(expr, valuation)
            for expr in symbolic_threads_range.num_work_items
        ]

        if symbolic_threads_range.dim < 3:
            threads_range += [1] * (3 - symbolic_threads_range.dim)

        def div_ceil(a, b):
            return a // b if a % b == 0 else a // b + 1

        #   TODO: Refine this?
        grid_size = tuple(
            div_ceil(threads, tpb)
            for threads, tpb in zip(threads_range, self._block_size)
        )
        assert len(grid_size) == 3

        launch_grid = LaunchGrid(grid_size, self._block_size)

        return tuple(args), launch_grid


class CupyJit(JitBase):

    def __init__(self, default_block_size: tuple[int, int, int] = (128, 2, 1)):
        self._runtime_headers = {"<cstdint>"}
        self._default_block_size = default_block_size

    def compile(self, kfunc: KernelFunction) -> Callable[..., None]:
        if not HAVE_CUPY:
            raise JitError(
                "`cupy` is not installed: just-in-time-compilation of CUDA kernels is unavailable."
            )

        if not isinstance(kfunc, GpuKernelFunction) or kfunc.target != Target.CUDA:
            raise ValueError(
                "The CupyJit just-in-time compiler only accepts kernels generated for CUDA or HIP"
            )

        options = self._compiler_options()
        prelude = self._prelude(kfunc)
        kernel_code = self._kernel_code(kfunc)
        code = prelude + kernel_code

        raw_kernel = cp.RawKernel(
            code, kfunc.name, options=options, backend="nvrtc", jitify=True
        )
        return CupyKernelWrapper(kfunc, raw_kernel, self._default_block_size)

    def _compiler_options(self) -> tuple[str, ...]:
        options = ["-w", "-std=c++11"]
        options.append("-I" + get_pystencils_include_path())
        return tuple(options)

    def _prelude(self, kfunc: GpuKernelFunction) -> str:
        headers = self._runtime_headers
        headers |= kfunc.required_headers

        if '"half_precision.h"' in headers:
            headers.remove('"half_precision.h"')
            if cp.cuda.runtime.is_hip:
                headers.add("<hip/hip_fp16.h>")
            else:
                headers.add("<cuda_fp16.h>")

        code = "\n".join(f"#include {header}" for header in headers)

        code += "\n\n#define RESTRICT __restrict__\n\n"

        return code

    def _kernel_code(self, kfunc: GpuKernelFunction) -> str:
        kernel_code = emit_code(kfunc)
        return f'extern "C" {kernel_code}'
