from __future__ import annotations
from typing import TYPE_CHECKING

from warnings import warn
from collections.abc import Collection

from typing import Sequence
from dataclasses import dataclass, InitVar, replace

from .target import Target
from .field import Field, FieldType

from .types import (
    PsIntegerType,
    UserTypeSpec,
    PsIeeeFloatType,
    PsScalarType,
    create_type,
)

from .defaults import DEFAULTS

if TYPE_CHECKING:
    from .backend.jit import JitBase


class PsOptionsError(Exception):
    """Indicates an option clash in the `CreateKernelConfig`."""


class _AUTO_TYPE:
    ...


AUTO = _AUTO_TYPE()
"""Special value that can be passed to some options for invoking automatic behaviour."""


@dataclass
class OpenMpConfig:
    """Parameters controlling kernel parallelization using OpenMP."""

    nesting_depth: int = 0
    """Nesting depth of the loop that should be parallelized. Must be a nonnegative number."""

    collapse: int = 0
    """Argument to the OpenMP ``collapse`` clause"""

    schedule: str = "static"
    """Argument to the OpenMP ``schedule`` clause"""

    num_threads: int | None = None
    """Set the number of OpenMP threads to execute the parallel region."""

    omit_parallel_construct: bool = False
    """If set to ``True``, the OpenMP ``parallel`` construct is omitted, producing just a ``#pragma omp for``.
    
    Use this option only if you intend to wrap the kernel into an external ``#pragma omp parallel`` region.
    """

    def __post_init__(self):
        if self.omit_parallel_construct and self.num_threads is not None:
            raise PsOptionsError(
                "Cannot specify `num_threads` if `omit_parallel_construct` is set."
            )


@dataclass
class CpuOptimConfig:
    """Configuration for the CPU optimizer.

    If any flag in this configuration is set to a value not supported by the CPU specified
    in `CreateKernelConfig.target`, an error will be raised.
    """

    openmp: bool | OpenMpConfig = False
    """Enable OpenMP parallelization.
    
    If set to `True`, the kernel will be parallelized using OpenMP according to the default settings in `OpenMpConfig`.
    To customize OpenMP parallelization, pass an instance of `OpenMpConfig` instead.
    """

    vectorize: bool | VectorizationConfig = False
    """Enable and configure auto-vectorization.
    
    If set to an instance of `VectorizationConfig` and a CPU target with vector capabilities is selected,
    pystencils will attempt to vectorize the kernel according to the given vectorization options.

    If set to `True`, pystencils will infer vectorization options from the given CPU target.

    If set to `False`, no vectorization takes place.
    """

    loop_blocking: None | tuple[int, ...] = None
    """Block sizes for loop blocking.
    
    If set, the kernel's loops will be tiled according to the given block sizes.
    """

    use_cacheline_zeroing: bool = False
    """Enable cache-line zeroing.
    
    If set to `True` and the selected CPU supports cacheline zeroing, the CPU optimizer will attempt
    to produce cacheline zeroing instructions where possible.
    """

    def get_vectorization_config(self) -> VectorizationConfig | None:
        if self.vectorize is True:
            return VectorizationConfig()
        elif isinstance(self.vectorize, VectorizationConfig):
            return self.vectorize
        else:
            return None


@dataclass
class VectorizationConfig:
    """Configuration for the auto-vectorizer.

    If any flag in this configuration is set to a value not supported by the CPU specified
    in `CreateKernelConfig.target`, an error will be raised.
    """

    lanes: int | None = None
    """Number of SIMD lanes to be used in vectorization.

    If set to `None` (the default), the vector register width will be automatically set to the broadest possible.
    
    If the CPU architecture specified in `target <CreateKernelConfig.target>` does not support some
    operation contained in the kernel with the given number of lanes, an error will be raised.
    """

    use_nontemporal_stores: bool | Collection[str | Field] = False
    """Enable nontemporal (streaming) stores.
    
    If set to `True` and the selected CPU supports streaming stores, the vectorizer will generate
    nontemporal store instructions for all stores.

    If set to a collection of fields (or field names), streaming stores will only be generated for
    the given fields.
    """

    assume_aligned: bool = False
    """Assume field pointer alignment.
    
    If set to `True`, the vectorizer will assume that the address of the first inner entry
    (after ghost layers) of each field is aligned at the necessary byte boundary.
    """

    assume_inner_stride_one: bool = False
    """Assume stride associated with the innermost spatial coordinate of all fields is one.
    
    If set to `True`, the vectorizer will replace the stride of the innermost spatial coordinate
    with unity, thus enabling vectorization. If any fields already have a fixed innermost stride
    that is not equal to one, an error will be raised.
    """

    @staticmethod
    def default_lanes(target: Target, dtype: PsScalarType):
        if not target.is_vector_cpu():
            raise ValueError(f"Given target {target} is no vector CPU target.")

        assert dtype.itemsize is not None

        match target:
            case Target.X86_SSE:
                return 128 // (dtype.itemsize * 8)
            case Target.X86_AVX:
                return 256 // (dtype.itemsize * 8)
            case Target.X86_AVX512 | Target.X86_AVX512_FP16:
                return 512 // (dtype.itemsize * 8)
            case _:
                raise NotImplementedError(
                    f"No default number of lanes known for {dtype} on {target}"
                )


@dataclass
class GpuIndexingConfig:
    """Configure index translation behaviour for kernels generated for GPU targets."""

    omit_range_check: bool = False
    """If set to `True`, omit the iteration counter range check.
    
    By default, the code generator introduces a check if the iteration counters computed from GPU block and thread
    indices are within the prescribed loop range.
    This check can be discarded through this option, at your own peril.
    """

    block_size: tuple[int, int, int] | None = None
    """Desired block size for the execution of GPU kernels. May be overridden later by the runtime system."""

    manual_launch_grid: bool = False
    """Always require a manually specified launch grid when running this kernel.
    
    If set to `True`, the code generator will not attempt to infer the size of
    the launch grid from the kernel.
    The launch grid will then have to be specified manually at runtime.
    """

    sycl_automatic_block_size: bool = True
    """If set to `True` while generating for `Target.SYCL`, let the SYCL runtime decide on the block size.

    If set to `True`, the kernel is generated for execution via
    `parallel_for <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_parallel_for_invoke>`_
    -dispatch using
    a flat ``sycl::range``. In this case, the GPU block size will be inferred by the SYCL runtime.

    If set to `False`, the kernel will receive an ``nd_item`` and has to be executed using
    `parallel_for <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_parallel_for_invoke>`_
    with an ``nd_range``. This allows manual specification of the block size.
    """


@dataclass
class CreateKernelConfig:
    """Options for create_kernel."""

    target: Target = Target.GenericCPU
    """The code generation target."""

    jit: JitBase | None = None
    """Just-in-time compiler used to compile and load the kernel for invocation from the current Python environment.
    
    If left at `None`, a default just-in-time compiler will be inferred from the `target` parameter.
    To explicitly disable JIT compilation, pass `pystencils.backend.jit.no_jit`.
    """

    function_name: str = "kernel"
    """Name of the generated function"""

    ghost_layers: None | _AUTO_TYPE | int | Sequence[int | tuple[int, int]] = None
    """Specifies the number of ghost layers of the iteration region.
    
    Options:
     - :py:data:`AUTO <pystencils.config.AUTO>`: Required ghost layers are inferred from field accesses
     - `int`:  A uniform number of ghost layers in each spatial coordinate is applied
     - ``Sequence[int, tuple[int, int]]``: Ghost layers are specified for each spatial coordinate.
        In each coordinate, a single integer specifies the ghost layers at both the lower and upper iteration limit,
        while a pair of integers specifies the lower and upper ghost layers separately.

    When manually specifying ghost layers, it is the user's responsibility to avoid out-of-bounds memory accesses.

    .. note::
        At most one of `ghost_layers`, `iteration_slice`, and `index_field` may be set.
    """

    iteration_slice: None | int | slice | tuple[int | slice] = None
    """Specifies the kernel's iteration slice.

    Example:
        >>> cfg = CreateKernelConfig(
        ...     iteration_slice=ps.make_slice[3:14, 2:-2]
        ... )
        >>> cfg.iteration_slice
        (slice(3, 14, None), slice(2, -2, None))

    .. note::
        At most one of `ghost_layers`, `iteration_slice`, and `index_field` may be set.
    """

    index_field: Field | None = None
    """Index field for a sparse kernel.
    
    If this option is set, a sparse kernel with the given field as index field will be generated.

    .. note::
        At most one of `ghost_layers`, `iteration_slice`, and `index_field` may be set.
    """

    """Data Types"""

    index_dtype: UserTypeSpec = DEFAULTS.index_dtype
    """Data type used for all index calculations."""

    default_dtype: UserTypeSpec = PsIeeeFloatType(64)
    """Default numeric data type.
    
    This data type will be applied to all untyped symbols.
    """

    """Analysis"""

    allow_double_writes: bool = False
    """
    If True, don't check if every field is only written at a single location. This is required
    for example for kernels that are compiled with loop step sizes > 1, that handle multiple
    cells at once. Use with care!
    """

    skip_independence_check: bool = False
    """
    By default the assignment list is checked for read/write independence. This means fields are only written at
    locations where they are read. Doing so guarantees thread safety. In some cases e.g. for
    periodicity kernel, this can not be assured and does the check needs to be deactivated. Use with care!
    """

    """Target-Specific Options"""

    cpu_optim: None | CpuOptimConfig = None
    """Configuration of the CPU kernel optimizer.
    
    If this parameter is set while `target` is a non-CPU target, an error will be raised.
    """

    gpu_indexing: None | GpuIndexingConfig = None
    """Configure index translation for GPU kernels.
    
    It this parameter is set while `target` is not a GPU target, an error will be raised.
    """

    #   Deprecated Options

    data_type: InitVar[UserTypeSpec | None] = None
    """Deprecated; use `default_dtype` instead"""

    cpu_openmp: InitVar[bool | int | None] = None
    """Deprecated; use `cpu_optim.openmp <CpuOptimConfig.openmp>` instead."""

    cpu_vectorize_info: InitVar[dict | None] = None
    """Deprecated; use `cpu_optim.vectorize <CpuOptimConfig.vectorize>` instead."""

    gpu_indexing_params: InitVar[dict | None] = None
    """Deprecated; use `gpu_indexing` instead."""

    #   Getters

    def get_target(self) -> Target:
        match self.target:
            case Target.CurrentCPU:
                return Target.auto_cpu()
            case _:
                return self.target

    def get_jit(self) -> JitBase:
        """Returns either the user-specified JIT compiler, or infers one from the target if none is given."""
        if self.jit is None:
            if self.target.is_cpu():
                from .backend.jit import LegacyCpuJit

                return LegacyCpuJit()
            elif self.target == Target.CUDA:
                try:
                    from .backend.jit.gpu_cupy import CupyJit

                    if (
                        self.gpu_indexing is not None
                        and self.gpu_indexing.block_size is not None
                    ):
                        return CupyJit(self.gpu_indexing.block_size)
                    else:
                        return CupyJit()

                except ImportError:
                    from .backend.jit import no_jit

                    return no_jit

            elif self.target == Target.SYCL:
                from .backend.jit import no_jit

                return no_jit
            else:
                raise NotImplementedError(
                    f"No default JIT compiler implemented yet for target {self.target}"
                )
        else:
            return self.jit

    #   Postprocessing

    def __post_init__(self, *args):

        #   Check deprecated options
        self._check_deprecations(*args)

        #   Check index data type
        if not isinstance(create_type(self.index_dtype), PsIntegerType):
            raise PsOptionsError("`index_dtype` was not an integer type.")

        #   Check iteration space argument consistency
        if (
            int(self.iteration_slice is not None)
            + int(self.ghost_layers is not None)
            + int(self.index_field is not None)
            > 1
        ):
            raise PsOptionsError(
                "Parameters `iteration_slice`, `ghost_layers` and 'index_field` are mutually exclusive; "
                "at most one of them may be set."
            )

        #   Check index field
        if (
            self.index_field is not None
            and self.index_field.field_type != FieldType.INDEXED
        ):
            raise PsOptionsError(
                "Only fields with `field_type == FieldType.INDEXED` can be specified as `index_field`"
            )

        #   Check optim
        if self.cpu_optim is not None:
            if (
                self.cpu_optim.vectorize is not False
                and not self.target.is_vector_cpu()
            ):
                raise PsOptionsError(
                    f"Cannot enable auto-vectorization for non-vector CPU target {self.target}"
                )

        if self.gpu_indexing is not None:
            if isinstance(self.gpu_indexing, str):
                match self.gpu_indexing:
                    case "block":
                        self.gpu_indexing = GpuIndexingConfig()
                    case "line":
                        raise NotImplementedError(
                            "GPU line indexing is currently unavailable."
                        )
                    case other:
                        raise PsOptionsError(
                            f"Invalid value for option gpu_indexing: {other}"
                        )

    def _check_deprecations(
        self,
        data_type: UserTypeSpec | None,
        cpu_openmp: bool | int | None,
        cpu_vectorize_info: dict | None,
        gpu_indexing_params: dict | None,
    ):
        optim: CpuOptimConfig | None = None

        if data_type is not None:
            _deprecated_option("data_type", "default_dtype")
            warn(
                "Setting the deprecated `data_type` will override the value of `default_dtype`. "
                "Set `default_dtype` instead.",
                UserWarning,
            )
            self.default_dtype = data_type

        if cpu_openmp is not None:
            _deprecated_option("cpu_openmp", "cpu_optim.openmp")

            match cpu_openmp:
                case True:
                    deprecated_omp = OpenMpConfig()
                case False:
                    deprecated_omp = False
                case int():
                    deprecated_omp = OpenMpConfig(num_threads=cpu_openmp)
                case _:
                    raise PsOptionsError(
                        f"Invalid option for `cpu_openmp`: {cpu_openmp}"
                    )

            optim = CpuOptimConfig(openmp=deprecated_omp)

        if cpu_vectorize_info is not None:
            _deprecated_option("cpu_vectorize_info", "cpu_optim.vectorize")
            if "instruction_set" in cpu_vectorize_info:
                if self.target != Target.GenericCPU:
                    raise PsOptionsError(
                        "Setting 'instruction_set' in the deprecated 'cpu_vectorize_info' option is only "
                        "valid if `target == Target.CPU`."
                    )

                isa = cpu_vectorize_info["instruction_set"]
                vec_target: Target
                match isa:
                    case "best":
                        vec_target = Target.available_vector_cpu_targets().pop()
                    case "sse":
                        vec_target = Target.X86_SSE
                    case "avx":
                        vec_target = Target.X86_AVX
                    case "avx512":
                        vec_target = Target.X86_AVX512
                    case "avx512vl":
                        vec_target = Target.X86_AVX512 | Target._VL
                    case _:
                        raise PsOptionsError(
                            f'Value {isa} in `cpu_vectorize_info["instruction_set"]` is not supported.'
                        )

                warn(
                    f"Value {isa} for `instruction_set` in deprecated `cpu_vectorize_info` "
                    "will override the `target` option. "
                    f"Set `target` to {vec_target} instead.",
                    UserWarning,
                )

                self.target = vec_target

            deprecated_vec_opts = VectorizationConfig(
                assume_inner_stride_one=cpu_vectorize_info.get(
                    "assume_inner_stride_one", False
                ),
                assume_aligned=cpu_vectorize_info.get("assume_aligned", False),
                use_nontemporal_stores=cpu_vectorize_info.get("nontemporal", False),
            )

            if optim is not None:
                optim = replace(optim, vectorize=deprecated_vec_opts)
            else:
                optim = CpuOptimConfig(vectorize=deprecated_vec_opts)

        if optim is not None:
            if self.cpu_optim is not None:
                raise PsOptionsError(
                    "Cannot specify both `cpu_optim` and a deprecated legacy optimization option at the same time."
                )
            else:
                self.cpu_optim = optim

        if gpu_indexing_params is not None:
            _deprecated_option("gpu_indexing_params", "gpu_indexing")

            if self.gpu_indexing is not None:
                raise PsOptionsError(
                    "Cannot specify both `gpu_indexing` and the deprecated `gpu_indexing_params` at the same time."
                )

            self.gpu_indexing = GpuIndexingConfig(
                block_size=gpu_indexing_params.get("block_size", None)
            )


def _deprecated_option(name, instead):
    from warnings import warn

    warn(
        f"The `{name}` option of CreateKernelConfig is deprecated and will be removed in pystencils 2.1. "
        f"Use `{instead}` instead.",
        FutureWarning,
    )
