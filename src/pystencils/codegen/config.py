from __future__ import annotations
from typing import TYPE_CHECKING

from warnings import warn
from abc import ABC
from collections.abc import Collection

from typing import Sequence, Generic, TypeVar, Callable, Any, cast
from dataclasses import dataclass, InitVar, fields

from .target import Target
from ..field import Field, FieldType

from ..types import (
    PsIntegerType,
    UserTypeSpec,
    PsScalarType,
    create_type,
)

from ..defaults import DEFAULTS

if TYPE_CHECKING:
    from ..jit import JitBase


Option_T = TypeVar("Option_T")
"""Type variable for option values"""


Arg_T = TypeVar("Arg_T")
"""Type variable for option arguments"""


class Option(Generic[Option_T, Arg_T]):
    """Option descriptor.

    This descriptor is used to model configuration options.
    It maintains a default value for the option that is used when no value
    was specified by the user.

    In configuration options, the value `None` stands for ``unset``.
    It can therefore not be used to set an option to the meaning "not any", or "empty"
    - for these, special values need to be used.

    The Option allows a validator function to be specified,
    which will be called to perform sanity checks on user-provided values.

    Through the validator, options may also be set from arguments of a different type (``Arg_T``)
    than their value type (``Option_T``). If ``Arg_T`` is different from ``Option_T``,
    the validator must perform the conversion from the former to the latter.

    .. note::
        ``Arg_T`` must always be a supertype of ``Option_T``.
    """

    def __init__(
        self,
        default: Option_T | None = None,
        validator: Callable[[Any, Arg_T | None], Option_T | None] | None = None,
    ) -> None:
        self._default = default
        self._validator = validator
        self._name: str
        self._lookup: str

    def validate(self, validator: Callable[[Any, Any], Any] | None):
        self._validator = validator
        return validator

    @property
    def default(self) -> Option_T | None:
        return self._default

    def get(self, obj) -> Option_T | None:
        val = getattr(obj, self._lookup, None)
        if val is None:
            return self._default
        else:
            return val

    def is_set(self, obj) -> bool:
        return getattr(obj, self._lookup, None) is not None

    def __set_name__(self, owner: ConfigBase, name: str):
        self._name = name
        self._lookup = f"_{name}"

    def __get__(self, obj: ConfigBase, objtype: type[ConfigBase] | None = None) -> Option_T | None:
        if obj is None:
            return None

        return getattr(obj, self._lookup, None)

    def __set__(self, obj: ConfigBase, arg: Arg_T | None):
        if arg is not None and self._validator is not None:
            value = self._validator(obj, arg)
        else:
            value = cast(Option_T, arg)
        setattr(obj, self._lookup, value)

    def __delete__(self, obj):
        delattr(obj, self._lookup)


class BasicOption(Option[Option_T, Option_T]):
    "Subclass of Option where ``Arg_T == Option_T``."


class ConfigBase(ABC):
    """Base class for configuration categories.

    This class implements query and retrieval mechanism for configuration options,
    as well as deepcopy functionality for categories.

    Subclasses of `ConfigBase` must be `dataclasses`,
    and all of their instance fields must have one of two descriptors types:
    - Either `Option`, for scalar options;
    - Or `Category` for option subcategories.

    `Option` fields must be assigned immutable values, but are otherwise unconstrained.
    `Category` subobjects must be subclasses of `ConfigBase`.

    **Retrieval** Options set to `None` are considered *unset*, i.e. the user has not provided a value.
    Through the `Option` descriptor, these options can still have a default value.
    To retrieve either the user-set value if one exists, or the default value otherwise, use `get_option`.

    **Deep-Copy** When a configuration object is copied, all of its subcategories must be copied along with it,
    such that changes in the original do no affect the copy, and vice versa.
    Such a deep copy is performed by the `copy <ConfigBase.copy>` method.
    """

    def get_option(self, name: str) -> Any:
        """Get the value set for the specified option, or the option's default value if none has been set."""
        descr: Option = type(self).__dict__[name]
        return descr.get(self)

    def is_option_set(self, name: str) -> bool:
        descr: Option = type(self).__dict__[name]
        return descr.is_set(self)

    def override(self, other: ConfigBase):
        for f in fields(self):  # type: ignore
            fvalue = getattr(self, f.name)
            if isinstance(fvalue, ConfigBase):  # type: ignore
                fvalue.override(getattr(other, f.name))
            else:
                new_val = getattr(other, f.name)
                if new_val is not None:
                    setattr(self, f.name, new_val)

    def copy(self):
        """Perform a semi-deep copy of this configuration object.

        This will recursively copy any config subobjects
        (categories, i.e. subclasses of `ConfigBase` wrapped in the `Category` descriptor)
        nested in this configuration object. Any other fields will be copied by reference.
        """

        #   IMPLEMENTATION NOTES
        #
        #   We do not need to call `copy` on any subcategories here, since the `Category`
        #   descriptor already calls `copy` in its `__set__` method,
        #   which is invoked during the constructor call in the `return` statement.
        #   Calling `copy` here would result in copying category objects twice.
        #
        #   We cannot use the standard library `copy.copy` here, since it merely duplicates
        #   the instance dictionary and does not call the constructor.

        config_fields = fields(self)  # type: ignore
        kwargs = dict()
        for field in config_fields:
            val = getattr(self, field.name)
            kwargs[field.name] = val
        return type(self)(**kwargs)


Category_T = TypeVar("Category_T", bound=ConfigBase)
"""Type variable for option categories."""


class Category(Generic[Category_T]):
    """Descriptor for a category of options.

    This descriptor makes sure that when an entire category is set to an object,
    that object is copied immediately such that later changes to the original
    do not affect this configuration.
    """

    def __init__(self, default: Category_T):
        self._default = default

    def __set_name__(self, owner: ConfigBase, name: str):
        self._name = name
        self._lookup = f"_{name}"

    def __get__(self, obj: ConfigBase, objtype: type[ConfigBase] | None = None) -> Category_T:
        if obj is None:
            return self._default

        return cast(Category_T, getattr(obj, self._lookup, None))

    def __set__(self, obj: ConfigBase, cat: Category_T):
        setattr(obj, self._lookup, cat.copy())


class _AUTO_TYPE: ...  # noqa: E701


AUTO = _AUTO_TYPE()
"""Special value that can be passed to some options for invoking automatic behaviour."""


@dataclass
class OpenMpOptions(ConfigBase):
    """Configuration options controlling automatic OpenMP instrumentation."""

    enable: BasicOption[bool] = BasicOption(False)
    """Enable OpenMP instrumentation"""

    nesting_depth: BasicOption[int] = BasicOption(0)
    """Nesting depth of the loop that should be parallelized. Must be a nonnegative number."""

    collapse: BasicOption[int] = BasicOption()
    """Argument to the OpenMP ``collapse`` clause"""

    schedule: BasicOption[str] = BasicOption("static")
    """Argument to the OpenMP ``schedule`` clause"""

    num_threads: BasicOption[int] = BasicOption()
    """Set the number of OpenMP threads to execute the parallel region."""

    omit_parallel_construct: BasicOption[bool] = BasicOption(False)
    """If set to ``True``, the OpenMP ``parallel`` construct is omitted, producing just a ``#pragma omp for``.
    
    Use this option only if you intend to wrap the kernel into an external ``#pragma omp parallel`` region.
    """


@dataclass
class VectorizationOptions(ConfigBase):
    """Configuration for the auto-vectorizer."""

    enable: BasicOption[bool] = BasicOption(False)
    """Enable intrinsic vectorization."""

    lanes: BasicOption[int] = BasicOption()
    """Number of SIMD lanes to be used in vectorization.

    If set to `None` (the default), the vector register width will be automatically set to the broadest possible.
    
    If the CPU architecture specified in `target <CreateKernelConfig.target>` does not support some
    operation contained in the kernel with the given number of lanes, an error will be raised.
    """

    use_nontemporal_stores: BasicOption[bool | Collection[str | Field]] = BasicOption(
        False
    )
    """Enable nontemporal (streaming) stores.
    
    If set to `True` and the selected CPU supports streaming stores, the vectorizer will generate
    nontemporal store instructions for all stores.

    If set to a collection of fields (or field names), streaming stores will only be generated for
    the given fields.
    """

    assume_aligned: BasicOption[bool] = BasicOption(False)
    """Assume field pointer alignment.
    
    If set to `True`, the vectorizer will assume that the address of the first inner entry
    (after ghost layers) of each field is aligned at the necessary byte boundary.
    """

    assume_inner_stride_one: BasicOption[bool] = BasicOption(False)
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
class CpuOptions(ConfigBase):
    """Configuration options specific to CPU targets."""

    openmp: Category[OpenMpOptions] = Category(OpenMpOptions())
    """Options governing OpenMP-instrumentation.
    """

    vectorize: Category[VectorizationOptions] = Category(VectorizationOptions())
    """Options governing intrinsic vectorization.
    """

    loop_blocking: BasicOption[tuple[int, ...]] = BasicOption()
    """Block sizes for loop blocking.
    
    If set, the kernel's loops will be tiled according to the given block sizes.
    """

    use_cacheline_zeroing: BasicOption[bool] = BasicOption(False)
    """Enable cache-line zeroing.
    
    If set to `True` and the selected CPU supports cacheline zeroing, the CPU optimizer will attempt
    to produce cacheline zeroing instructions where possible.
    """


@dataclass
class GpuOptions(ConfigBase):
    """Configuration options specific to GPU targets."""

    omit_range_check: BasicOption[bool] = BasicOption(False)
    """If set to `True`, omit the iteration counter range check.
    
    By default, the code generator introduces a check if the iteration counters computed from GPU block and thread
    indices are within the prescribed loop range.
    This check can be discarded through this option, at your own peril.
    """

    block_size: BasicOption[tuple[int, int, int]] = BasicOption()
    """Desired block size for the execution of GPU kernels. May be overridden later by the runtime system."""

    manual_launch_grid: BasicOption[bool] = BasicOption(False)
    """Always require a manually specified launch grid when running this kernel.
    
    If set to `True`, the code generator will not attempt to infer the size of
    the launch grid from the kernel.
    The launch grid will then have to be specified manually at runtime.
    """


@dataclass
class SyclOptions(ConfigBase):
    """Options specific to the `SYCL <Target.SYCL>` target."""

    automatic_block_size: BasicOption[bool] = BasicOption(True)
    """If set to `True`, let the SYCL runtime decide on the block size.

    If set to `True`, the kernel is generated for execution via
    `parallel_for <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_parallel_for_invoke>`_
    -dispatch using
    a flat ``sycl::range``. In this case, the GPU block size will be inferred by the SYCL runtime.

    If set to `False`, the kernel will receive an ``nd_item`` and has to be executed using
    `parallel_for <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_parallel_for_invoke>`_
    with an ``nd_range``. This allows manual specification of the block size.
    """


GhostLayerSpec = _AUTO_TYPE | int | Sequence[int | tuple[int, int]]


IterationSliceSpec = int | slice | tuple[int | slice]


@dataclass
class CreateKernelConfig(ConfigBase):
    """Options for create_kernel."""

    target: BasicOption[Target] = BasicOption(Target.GenericCPU)
    """The code generation target."""

    jit: BasicOption[JitBase] = BasicOption()
    """Just-in-time compiler used to compile and load the kernel for invocation from the current Python environment.
    
    If left at `None`, a default just-in-time compiler will be inferred from the `target` parameter.
    To explicitly disable JIT compilation, pass `pystencils.no_jit <pystencils.jit.no_jit>`.
    """

    function_name: BasicOption[str] = BasicOption("kernel")
    """Name of the generated function"""

    ghost_layers: BasicOption[GhostLayerSpec] = BasicOption()
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

    iteration_slice: BasicOption[IterationSliceSpec] = BasicOption()
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

    index_field: BasicOption[Field] = BasicOption()
    """Index field for a sparse kernel.
    
    If this option is set, a sparse kernel with the given field as index field will be generated.

    .. note::
        At most one of `ghost_layers`, `iteration_slice`, and `index_field` may be set.
    """

    """Data Types"""

    index_dtype: Option[PsIntegerType, UserTypeSpec] = Option(DEFAULTS.index_dtype)
    """Data type used for all index calculations."""

    default_dtype: Option[PsScalarType, UserTypeSpec] = Option(DEFAULTS.numeric_dtype)
    """Default numeric data type.
    
    This data type will be applied to all untyped symbols.
    """

    """Analysis"""

    allow_double_writes: BasicOption[bool] = BasicOption(False)
    """
    If True, don't check if every field is only written at a single location. This is required
    for example for kernels that are compiled with loop step sizes > 1, that handle multiple
    cells at once. Use with care!
    """

    skip_independence_check: BasicOption[bool] = BasicOption(False)
    """
    By default the assignment list is checked for read/write independence. This means fields are only written at
    locations where they are read. Doing so guarantees thread safety. In some cases e.g. for
    periodicity kernel, this can not be assured and does the check needs to be deactivated. Use with care!
    """

    """Target-Specific Options"""

    cpu: Category[CpuOptions] = Category(CpuOptions())
    """Options for CPU kernels. See `CpuOptions`."""

    gpu: Category[GpuOptions] = Category(GpuOptions())
    """Options for GPU Kernels. See `GpuOptions`."""

    sycl: Category[SyclOptions] = Category(SyclOptions())
    """Options for SYCL kernels. See `SyclOptions`."""

    @index_dtype.validate
    def validate_index_type(self, spec: UserTypeSpec):
        dtype = create_type(spec)
        if not isinstance(dtype, PsIntegerType):
            raise ValueError("index_dtype must be an integer type")
        return dtype

    @default_dtype.validate
    def validate_default_dtype(self, spec: UserTypeSpec):
        dtype = create_type(spec)
        if not isinstance(dtype, PsScalarType):
            raise ValueError("default_dtype must be a scalar numeric type")
        return dtype

    @index_field.validate
    def validate_index_field(self, idx_field: Field):
        if idx_field.field_type != FieldType.INDEXED:
            raise ValueError(
                "Only fields of type FieldType.INDEXED can be used as index fields"
            )
        return idx_field

    #   Deprecated Options

    data_type: InitVar[UserTypeSpec | None] = None
    """Deprecated; use `default_dtype` instead"""

    cpu_openmp: InitVar[bool | int | None] = None
    """Deprecated; use `cpu.openmp <CpuOptions.openmp>` instead."""

    cpu_vectorize_info: InitVar[dict | None] = None
    """Deprecated; use `cpu.vectorize <CpuOptions.vectorize>` instead."""

    gpu_indexing_params: InitVar[dict | None] = None
    """Deprecated; set options in the `gpu` category instead."""

    #   Getters

    def get_target(self) -> Target:
        t: Target = self.get_option("target")
        match t:
            case Target.CurrentCPU:
                return Target.auto_cpu()
            case _:
                return t

    def get_jit(self) -> JitBase:
        """Returns either the user-specified JIT compiler, or infers one from the target if none is given."""
        jit: JitBase | None = self.get_option("jit")

        if jit is None:
            if self.get_target().is_cpu():
                from ..jit import LegacyCpuJit

                return LegacyCpuJit()
            elif self.get_target() == Target.CUDA:
                try:
                    from ..jit.gpu_cupy import CupyJit

                    if self.gpu is not None and self.gpu.block_size is not None:
                        return CupyJit(self.gpu.block_size)
                    else:
                        return CupyJit()

                except ImportError:
                    from ..jit import no_jit

                    return no_jit

            elif self.get_target() == Target.SYCL:
                from ..jit import no_jit

                return no_jit
            else:
                raise NotImplementedError(
                    f"No default JIT compiler implemented yet for target {self.target}"
                )
        else:
            return jit

    #   Postprocessing

    def __post_init__(self, *args):
        #   Check deprecated options
        self._check_deprecations(*args)

    def _check_deprecations(
        self,
        data_type: UserTypeSpec | None,
        cpu_openmp: bool | int | None,
        cpu_vectorize_info: dict | None,
        gpu_indexing_params: dict | None,
    ):  # pragma: no cover
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
            warn(
                "Setting the deprecated `cpu_openmp` option will override any options "
                "passed in the `cpu.openmp` category.",
                UserWarning,
            )

            deprecated_omp = OpenMpOptions()
            match cpu_openmp:
                case True:
                    deprecated_omp.enable = False
                case False:
                    deprecated_omp.enable = False
                case int():
                    deprecated_omp.enable = True
                    deprecated_omp.num_threads = cpu_openmp
                case _:
                    raise ValueError(
                        f"Invalid option for `cpu_openmp`: {cpu_openmp}"
                    )

            self.cpu.openmp = deprecated_omp

        if cpu_vectorize_info is not None:
            _deprecated_option("cpu_vectorize_info", "cpu_optim.vectorize")
            if "instruction_set" in cpu_vectorize_info:
                if self.target != Target.GenericCPU:
                    raise ValueError(
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
                        raise ValueError(
                            f'Value {isa} in `cpu_vectorize_info["instruction_set"]` is not supported.'
                        )

                warn(
                    f"Value {isa} for `instruction_set` in deprecated `cpu_vectorize_info` "
                    "will override the `target` option. "
                    f"Set `target` to {vec_target} instead.",
                    UserWarning,
                )

                self.target = vec_target

            warn(
                "Setting the deprecated `cpu_vectorize_info` will override any options "
                "passed in the `cpu.vectorize` category.",
                UserWarning,
            )

            deprecated_vec_opts = VectorizationOptions(
                enable=True,
                assume_inner_stride_one=cpu_vectorize_info.get(
                    "assume_inner_stride_one", False
                ),
                assume_aligned=cpu_vectorize_info.get("assume_aligned", False),
                use_nontemporal_stores=cpu_vectorize_info.get("nontemporal", False),
            )

            self.cpu.vectorize = deprecated_vec_opts

        if gpu_indexing_params is not None:
            _deprecated_option("gpu_indexing_params", "gpu_indexing")
            warn(
                "Setting the deprecated `gpu_indexing_params` will override any options "
                "passed in the `gpu` category."
            )

            self.gpu = GpuOptions(
                block_size=gpu_indexing_params.get("block_size", None)
            )


def _deprecated_option(name, instead):  # pragma: no cover
    from warnings import warn

    warn(
        f"The `{name}` option of CreateKernelConfig is deprecated and will be removed in pystencils 2.1. "
        f"Use `{instead}` instead.",
        FutureWarning,
    )
