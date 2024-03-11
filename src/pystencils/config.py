from __future__ import annotations

from collections.abc import Collection

from typing import Sequence
from dataclasses import dataclass

from .enums import Target
from .field import Field, FieldType

from .backend.jit import JitBase
from .backend.exceptions import PsOptionsError
from .types import PsIntegerType, PsNumericType, PsIeeeFloatType

from .defaults import DEFAULTS


@dataclass
class CpuOptimConfig:
    """Configuration for the CPU optimizer.
    
    If any flag in this configuration is set to a value not supported by the CPU specified
    in `CreateKernelConfig.target`, an error will be raised.
    """
    
    openmp: bool = False
    """Enable OpenMP parallelization.
    
    If set to `True`, the kernel will be parallelized using OpenMP according to the OpenMP settings
    given in this configuration.
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


@dataclass
class VectorizationConfig:
    """Configuration for the auto-vectorizer.
    
    If any flag in this configuration is set to a value not supported by the CPU specified
    in `CreateKernelConfig.target`, an error will be raised.
    """

    vector_width: int | None = None
    """Desired vector register width in bits.
    
    If set to an integer value, the vectorizer will use this as the desired vector register width.

    If set to `None`, the vector register width will be automatically set to the broadest possible.
    
    If the selected CPU does not support the given width, an error will be raised.
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


@dataclass
class CreateKernelConfig:
    """Options for create_kernel."""

    target: Target = Target.GenericCPU
    """The code generation target."""

    jit: JitBase | None = None
    """Just-in-time compiler used to compile and load the kernel for invocation from the current Python environment.
    
    If left at `None`, a default just-in-time compiler will be inferred from the `target` parameter.
    To explicitly disable JIT compilation, pass `pystencils.nbackend.jit.no_jit`.
    """

    function_name: str = "kernel"
    """Name of the generated function"""

    ghost_layers: None | int | Sequence[int | tuple[int, int]] = None
    """Specifies the number of ghost layers of the iteration region.
    
    Options:
     - `None`: Required ghost layers are inferred from field accesses
     - `int`:  A uniform number of ghost layers in each spatial coordinate is applied
     - ``Sequence[int, tuple[int, int]]``: Ghost layers are specified for each spatial coordinate.
        In each coordinate, a single integer specifies the ghost layers at both the lower and upper iteration limit,
        while a pair of integers specifies the lower and upper ghost layers separately.

    When manually specifying ghost layers, it is the user's responsibility to avoid out-of-bounds memory accesses.
    If ``ghost_layers=None`` is specified, the iteration region may otherwise be set using the `iteration_slice` option.
    """

    iteration_slice: None | Sequence[slice] = None
    """Specifies the kernel's iteration slice.
    
    `iteration_slice` may only be set if ``ghost_layers=None``.
    If it is set, a slice must be specified for each spatial coordinate.
    TODO: Specification of valid slices and their behaviour
    """

    index_field: Field | None = None
    """Index field for a sparse kernel.
    
    If this option is set, a sparse kernel with the given field as index field will be generated.
    """

    """Data Types"""

    index_dtype: PsIntegerType = DEFAULTS.index_dtype
    """Data type used for all index calculations."""

    default_dtype: PsNumericType = PsIeeeFloatType(64)
    """Default numeric data type.
    
    This data type will be applied to all untyped symbols.
    """

    cpu_optim: None | CpuOptimConfig = None
    """Configuration of the CPU kernel optimizer.
    
    If this parameter is set while `target` is a non-CPU target, an error will be raised.
    """

    def __post_init__(self):
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
            if not self.target.is_cpu():
                raise PsOptionsError(f"`cpu_optim` cannot be set for non-CPU target {self.target}")
            
            if self.cpu_optim.vectorize is not False and not self.target.is_vector_cpu():
                raise PsOptionsError(f"Cannot enable auto-vectorization for non-vector CPU target {self.target}")

        #   Infer JIT
        if self.jit is None:
            if self.target.is_cpu():
                from .backend.jit import LegacyCpuJit
                self.jit = LegacyCpuJit()
            else:
                raise NotImplementedError(
                    f"No default JIT compiler implemented yet for target {self.target}"
                )
