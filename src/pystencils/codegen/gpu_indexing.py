from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast, Any, Callable
from itertools import chain
from warnings import warn

from .functions import Lambda
from .parameters import Parameter
from .errors import CodegenError
from .config import GpuIndexingScheme
from .target import Target

from ..backend.kernelcreation import (
    KernelCreationContext,
    FullIterationSpace,
    SparseIterationSpace,
)
from ..backend.platforms.generic_gpu import ThreadMapping

from ..backend.ast.expressions import PsExpression, PsIntDiv
from math import prod

from ..utils import ceil_to_multiple

dim3 = tuple[int, int, int]
_Dim3Lambda = tuple[Lambda, Lambda, Lambda]


@dataclass
class HardwareProperties:
    warp_size: int | None
    max_threads_per_block: int
    max_block_sizes: dim3

    def block_size_exceeds_hw_limits(self, block_size: tuple[int, ...]) -> bool:
        """Checks if provided block size conforms limits given by the hardware."""

        return (
            any(
                size > max_size
                for size, max_size in zip(block_size, self.max_block_sizes)
            )
            or prod(block_size) > self.max_threads_per_block
        )


class GpuLaunchConfiguration(ABC):
    """Base class for launch configurations for CUDA and HIP kernels.

    Args:
        block_size: A triple of lambdas determining the GPU block size
        grid_size: A triple of lambdas determining the GPU grid size
        config_parameters: Set containing all parameters to the given lambdas that are not also
            parameters to the associated kernel
    """

    @property
    @abstractmethod
    def block_size(self) -> dim3 | None:
        """Returns desired block size if available."""
        pass

    @block_size.setter
    @abstractmethod
    def block_size(self, val: dim3):
        """Sets desired block size if possible."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> frozenset[Parameter]:
        """Parameters of this launch configuration"""

    @abstractmethod
    def evaluate(self, **kwargs) -> tuple[dim3, dim3]:
        """Compute block and grid size for a kernel launch.

        Args:
            kwargs: Valuation providing a value for each parameter listed in `parameters`
        """

    @abstractmethod
    def jit_cache_key(self) -> Any:
        """Return a hashable object that represents any user-configurable options of
        this launch configuration, such that when the configuration changes, the JIT parameter
        cache is invalidated."""

    @staticmethod
    def get_default_block_size(rank: int) -> dim3:
        """Returns the default block size configuration used by the generator."""

        match rank:
            case 1:
                return (256, 1, 1)
            case 2:
                return (16, 16, 1)
            case 3:
                return (8, 8, 4)
            case _:
                assert False, "unreachable code"

    @staticmethod
    def _excessive_block_size_error_msg(block_size: tuple[int, ...]):
        return (
            "Unable to determine GPU block size for this kernel. "
            f"Final block size was too large: {block_size}."
        )


class AutomaticLaunchConfiguration(GpuLaunchConfiguration):
    """Launch configuration that is dynamically computed from kernel parameters.

    This launch configuration permits no further user customization.
    """

    def __init__(
        self,
        block_size: _Dim3Lambda,
        grid_size: _Dim3Lambda,
        hw_props: HardwareProperties,
        assume_warp_aligned_block_size: bool,
    ) -> None:
        self._block_size = block_size
        self._grid_size = grid_size
        self._hw_props = hw_props
        self._assume_warp_aligned_block_size = assume_warp_aligned_block_size

        self._params: frozenset[Parameter] = frozenset().union(
            *(lb.parameters for lb in chain(block_size, grid_size))
        )

    @property
    def block_size(self) -> dim3 | None:
        """Block size is only available when `evaluate` is called."""
        return None

    @block_size.setter
    def block_size(self, val: dim3):
        AttributeError(
            "Setting `block_size` on an automatic launch configuration has no effect."
        )

    @property
    def parameters(self) -> frozenset[Parameter]:
        return self._params

    def evaluate(self, **kwargs) -> tuple[dim3, dim3]:
        block_size = tuple(int(bs(**kwargs)) for bs in self._block_size)

        if self._hw_props.block_size_exceeds_hw_limits(block_size):
            raise CodegenError(f"Block size {block_size} exceeds hardware limits.")

        grid_size = tuple(int(gs(**kwargs)) for gs in self._grid_size)
        return cast(dim3, block_size), cast(dim3, grid_size)

    def jit_cache_key(self) -> Any:
        return ()


class ManualLaunchConfiguration(GpuLaunchConfiguration):
    """Manual GPU launch configuration.

    This launch configuration requires the user to set block and grid size.
    """

    def __init__(
        self, hw_props: HardwareProperties, assume_warp_aligned_block_size: bool = False
    ) -> None:
        self._assume_warp_aligned_block_size = assume_warp_aligned_block_size

        self._hw_props = hw_props

        self._block_size: dim3 | None = None
        self._grid_size: dim3 | None = None

    @property
    def block_size(self) -> dim3 | None:
        return self._block_size

    @block_size.setter
    def block_size(self, val: dim3):
        self._block_size = val

    @property
    def grid_size(self) -> dim3 | None:
        return self._grid_size

    @grid_size.setter
    def grid_size(self, val: dim3):
        self._grid_size = val

    @property
    def parameters(self) -> frozenset[Parameter]:
        return frozenset()

    def evaluate(self, **kwargs) -> tuple[dim3, dim3]:
        if self._block_size is None:
            raise AttributeError("No GPU block size was set by the user.")

        if self._grid_size is None:
            raise AttributeError("No GPU grid size was set by the user.")

        if (
            self._assume_warp_aligned_block_size
            and self._hw_props.warp_size is not None
            and prod(self._block_size) % self._hw_props.warp_size != 0
        ):
            raise CodegenError(
                "Specified block sizes must align with warp size with "
                "`assume_warp_aligned_block_size` enabled."
            )

        if self._hw_props.block_size_exceeds_hw_limits(self._block_size):
            raise CodegenError(self._excessive_block_size_error_msg(self._block_size))

        return self._block_size, self._grid_size

    def jit_cache_key(self) -> Any:
        return (self._block_size, self._grid_size)


class DynamicBlockSizeLaunchConfiguration(GpuLaunchConfiguration):
    """GPU launch configuration that dynamically computes the grid size from either the default block size
    or a computed block size. Computing block sizes can be triggerred via the :meth:`trim_block_size` or
    :meth:`fit_block_size` member functions. These functions adapt a user-defined initial block size that they
    receive as an argument. The adaptation of the initial block sizes is described in the following:

    For each dimension :math:`c \\in \\{ x, y, z \\}`,

    - if :meth:`fit_block_size` was chosen:

        the initial block size is adapted such that it aligns with multiples of the hardware's warp size.
        This is done using a fitting algorithm first trims the initial block size with the iteration space
        and increases it incrementally until it is large enough and coincides with multiples of the warp size, i.e.

        ``block_size.c = _fit_block_size_to_it_space(iter_space.c, init_block_size.c, hardware_properties)``

        The fitted block size also guarantees the user usage of `GpuOptions.assume_warp_aligned_block_size`.

    - elif :meth:`trim_block_size` was chosen:

        a trimming between the number of work items and the kernel's iteration space occurs, i.e.

        - if ``init_block_size.c > num_work_items.c``, ``block_size = num_work_items.c``
        - otherwise, ``block_size.c = init_block_size.c``

        When `GpuOptions.assume_warp_aligned_block_size` is set, we ensure warp-alignment by
        rounding the block size dimension that is closest the next multiple of the warp size.

    - otherwise: the default block size is taken i.e.

        ``block_size.c = get_default_block_size(rank=3).c``

    The actual launch grid size is then computed as follows.

    ``grid_size.c = ceil(num_work_items.c / block_size.c)``.
    """

    def __init__(
        self,
        num_work_items: _Dim3Lambda,
        hw_props: HardwareProperties,
        assume_warp_aligned_block_size: bool,
    ) -> None:
        self._num_work_items = num_work_items

        self._hw_props = hw_props

        self._assume_warp_aligned_block_size = assume_warp_aligned_block_size

        default_bs = GpuLaunchConfiguration.get_default_block_size(len(num_work_items))
        self._default_block_size = default_bs
        self._init_block_size: dim3 = default_bs
        self._compute_block_size: (
            Callable[[dim3, dim3, HardwareProperties], tuple[int, ...]] | None
        ) = None

        self._params: frozenset[Parameter] = frozenset().union(
            *(wit.parameters for wit in num_work_items)
        )

    @property
    def num_work_items(self) -> _Dim3Lambda:
        """Lambda expressions that compute the number of work items in each iteration space
        dimension from kernel parameters."""
        return self._num_work_items

    @property
    def parameters(self) -> frozenset[Parameter]:
        """Parameters of this launch configuration"""
        return self._params
    
    @property
    def default_block_size(self) -> dim3:
        return self._default_block_size

    @property
    def block_size(self) -> dim3 | None:
        """Block size is only available when `evaluate` is called."""
        return None

    @block_size.setter
    def block_size(self, val: dim3):
        AttributeError(
            "Setting `block_size` on an dynamic launch configuration has no effect."
        )

    @staticmethod
    def _round_block_sizes_to_warp_size(
        to_round: tuple[int, ...], warp_size: int
    ) -> tuple[int, ...]:
        # check if already aligns with warp size
        if prod(to_round) % warp_size == 0:
            return tuple(to_round)

        # find index of element closest to warp size and round up
        index_to_round = to_round.index(max(to_round, key=lambda i: abs(i % warp_size)))
        if index_to_round + 1 < len(to_round):
            return (
                *to_round[:index_to_round],
                ceil_to_multiple(to_round[index_to_round], warp_size),
                *to_round[index_to_round + 1:],
            )
        else:
            return (
                *to_round[:index_to_round],
                ceil_to_multiple(to_round[index_to_round], warp_size),
            )

    def trim_block_size(self, block_size: dim3):
        def call_trimming_factory(
            it: dim3,
            bs: dim3,
            hw: HardwareProperties,
        ):
            return self._trim_block_size_to_it_space(it, bs, hw)

        self._init_block_size = block_size
        self._compute_block_size = call_trimming_factory

    def _trim_block_size_to_it_space(
        self,
        it_space: dim3,
        block_size: dim3,
        hw_props: HardwareProperties,
    ) -> tuple[int, ...]:
        """Returns specified block sizes trimmed with iteration space.
        Raises CodegenError if trimmed block size does not conform hardware limits.
        """

        ret = tuple([min(b, i) for b, i in zip(block_size, it_space)])
        if hw_props.block_size_exceeds_hw_limits(ret):
            raise CodegenError(self._excessive_block_size_error_msg(ret))

        if (
            self._assume_warp_aligned_block_size
            and hw_props.warp_size is not None
            and prod(ret) % hw_props.warp_size != 0
        ):
            self._round_block_sizes_to_warp_size(ret, hw_props.warp_size)

        return ret

    def fit_block_size(self, block_size: dim3):
        def call_fitting_factory(
            it: dim3,
            bs: dim3,
            hw: HardwareProperties,
        ):
            return self._fit_block_size_to_it_space(it, bs, hw)

        self._init_block_size = block_size
        self._compute_block_size = call_fitting_factory

    def _fit_block_size_to_it_space(
        self,
        it_space: dim3,
        block_size: dim3,
        hw_props: HardwareProperties,
    ) -> tuple[int, ...]:
        """Returns an optimized block size configuration with block sizes being aligned with the warp size.
        Raises CodegenError if optimal block size could not be found or does not conform hardware limits.
        """

        def trim(to_trim: list[int]) -> list[int]:
            return [min(b, i) for b, i in zip(to_trim, it_space)]

        def check_sizes_and_return(ret: tuple[int, ...]) -> tuple[int, ...]:
            if hw_props.block_size_exceeds_hw_limits(ret):
                raise CodegenError(self._excessive_block_size_error_msg(ret))
            return ret

        trimmed = trim(list(block_size))

        if hw_props.warp_size is None:
            return tuple(trimmed)

        if (
            prod(trimmed) >= hw_props.warp_size
            and prod(trimmed) % hw_props.warp_size == 0
        ):
            # case 1: greater than min block size -> use trimmed result
            return check_sizes_and_return(tuple(trimmed))

        prev_trim_size = 0
        resize_order = [0, 2, 1] if len(it_space) == 3 else range(len(it_space))
        while prod(trimmed) is not prev_trim_size:
            prev_trim_size = prod(trimmed)

            # case 2: trimmed block is equivalent to the whole iteration space
            if all(b == i for b, i in zip(trimmed, it_space)):
                return check_sizes_and_return(
                    self._round_block_sizes_to_warp_size(
                        tuple(trimmed), hw_props.warp_size
                    )
                )
            else:
                # double block size in each dimension until block is large enough (or case 2 triggers)
                for d in resize_order:
                    trimmed[d] *= 2

                    # trim fastest moving dim to multiples of warp size
                    if (
                        d == 0
                        and trimmed[d] > hw_props.warp_size
                        and trimmed[d] % hw_props.warp_size != 0
                    ):
                        # subtract remainder
                        trimmed[d] = trimmed[d] - (trimmed[d] % hw_props.warp_size)

                    # check if block sizes are within hardware capabilities
                    trimmed[d] = min(trimmed[d], hw_props.max_block_sizes[d])

                    # trim again
                    trimmed = trim(trimmed)

                    # case 3: trim block is large enough
                    if prod(trimmed) >= hw_props.warp_size:
                        return check_sizes_and_return(
                            self._round_block_sizes_to_warp_size(
                                tuple(trimmed), hw_props.warp_size
                            )
                        )

        raise CodegenError("Unable to determine GPU block size for this kernel.")

    def evaluate(self, **kwargs) -> tuple[dim3, dim3]:
        from ..utils import div_ceil

        num_work_items = cast(
            dim3, tuple(int(wit(**kwargs)) for wit in self._num_work_items)
        )

        block_size: dim3
        if self._compute_block_size:
            try:
                computed_bs = self._compute_block_size(
                    num_work_items, self._init_block_size, self._hw_props
                )

                block_size = cast(dim3, computed_bs)
            except CodegenError as e:
                block_size = self._default_block_size
                warn(
                    f"CodeGenError occurred: {getattr(e, 'message', repr(e))}. "
                    f"Block size fitting could not determine optimal block size configuration. "
                    f"Defaulting back to {self._default_block_size}."
                )
        else:
            block_size = self._default_block_size

        grid_size = cast(
            dim3,
            tuple(div_ceil(wit, bs) for wit, bs in zip(num_work_items, block_size)),
        )

        return block_size, grid_size

    def jit_cache_key(self) -> Any:
        return ()


class GpuIndexing:
    """Factory for GPU indexing objects required during code generation.

    This class acts as a helper class for the code generation driver.
    It produces both the `ThreadMapping` required by the backend,
    as well as factories for the launch configuration required later by the runtime system.

    Args:
        ctx: The kernel creation context
        scheme: The desired GPU indexing scheme
        block_size: A user-defined default block size, required only if the indexing scheme permits
            modification of the block size
        manual_launch_grid: If `True`, always emit a `ManualLaunchConfiguration` to force the runtime system
            to set the launch configuration explicitly
    """

    def __init__(
        self,
        ctx: KernelCreationContext,
        target: Target,
        scheme: GpuIndexingScheme,
        warp_size: int | None,
        manual_launch_grid: bool = False,
        assume_warp_aligned_block_size: bool = False,
    ) -> None:
        self._ctx = ctx
        self._target = target
        self._scheme = scheme
        self._manual_launch_grid = manual_launch_grid
        self._assume_warp_aligned_block_size = assume_warp_aligned_block_size

        self._hw_props = HardwareProperties(
            warp_size,
            self.get_max_threads_per_block(target),
            self.get_max_block_sizes(target),
        )

        from ..backend.kernelcreation import AstFactory
        from .driver import KernelFactory

        self._ast_factory = AstFactory(self._ctx)
        self._kernel_factory = KernelFactory(self._ctx)

    @staticmethod
    def get_max_block_sizes(target: Target):
        match target:
            case Target.CUDA:
                return (1024, 1024, 64)
            case Target.HIP:
                return (1024, 1024, 1024)
            case _:
                raise CodegenError(
                    f"Cannot determine max GPU block sizes for target {target}"
                )

    @staticmethod
    def get_max_threads_per_block(target: Target):
        match target:
            case Target.CUDA | Target.HIP:
                return 1024
            case _:
                raise CodegenError(
                    f"Cannot determine max GPU threads per block for target {target}"
                )

    def get_thread_mapping(self) -> ThreadMapping:
        """Retrieve a thread mapping object for use by the backend"""

        from ..backend.platforms.generic_gpu import Linear3DMapping, Blockwise4DMapping

        match self._scheme:
            case GpuIndexingScheme.Linear3D:
                return Linear3DMapping()
            case GpuIndexingScheme.Blockwise4D:
                return Blockwise4DMapping()

    def get_launch_config_factory(self) -> Callable[[], GpuLaunchConfiguration]:
        """Retrieve a factory for the launch configuration for later consumption by the runtime system"""
        if self._manual_launch_grid:

            def factory():
                return ManualLaunchConfiguration(
                    self._hw_props, self._assume_warp_aligned_block_size
                )

            return factory

        match self._scheme:
            case GpuIndexingScheme.Linear3D:
                return self._get_linear3d_config_factory()
            case GpuIndexingScheme.Blockwise4D:
                return self._get_blockwise4d_config_factory()

    def _get_linear3d_config_factory(
        self,
    ) -> Callable[[], DynamicBlockSizeLaunchConfiguration]:
        work_items_expr = self._get_work_items()
        rank = len(work_items_expr)

        if rank > 3:
            raise CodegenError(
                "Cannot create a launch grid configuration using the Linear3D indexing scheme"
                f" for a {rank}-dimensional kernel."
            )

        work_items_expr += tuple(
            self._ast_factory.parse_index(1) for _ in range(3 - rank)
        )

        num_work_items = cast(
            _Dim3Lambda,
            tuple(self._kernel_factory.create_lambda(wit) for wit in work_items_expr),
        )

        def factory():
            return DynamicBlockSizeLaunchConfiguration(
                num_work_items,
                self._hw_props,
                self._assume_warp_aligned_block_size,
            )

        return factory

    def _get_blockwise4d_config_factory(
        self,
    ) -> Callable[[], AutomaticLaunchConfiguration]:
        work_items = self._get_work_items()
        rank = len(work_items)

        if rank > 4:
            raise ValueError(f"Iteration space rank is too large: {rank}")

        # impossible to use block size determination function since the iteration space is unknown
        # -> round block size in fastest moving dimension up to multiple of warp size
        rounded_block_size: PsExpression
        if (
            self._assume_warp_aligned_block_size
            and self._hw_props.warp_size is not None
        ):
            warp_size = self._ast_factory.parse_index(self._hw_props.warp_size)
            rounded_block_size = self._ast_factory.parse_index(
                PsIntDiv(
                    work_items[0].clone()
                    + warp_size.clone()
                    - self._ast_factory.parse_index(1),
                    warp_size.clone(),
                )
                * warp_size.clone()
            )
        else:
            rounded_block_size = work_items[0]

        block_size = (
            self._kernel_factory.create_lambda(rounded_block_size),
            self._kernel_factory.create_lambda(self._ast_factory.parse_index(1)),
            self._kernel_factory.create_lambda(self._ast_factory.parse_index(1)),
        )

        grid_size = tuple(
            self._kernel_factory.create_lambda(wit) for wit in work_items[1:]
        ) + tuple(
            self._kernel_factory.create_lambda(self._ast_factory.parse_index(1))
            for _ in range(4 - rank)
        )

        def factory():
            return AutomaticLaunchConfiguration(
                block_size,
                cast(_Dim3Lambda, grid_size),
                self._hw_props,
                self._assume_warp_aligned_block_size,
            )

        return factory

    def _get_work_items(self) -> tuple[PsExpression, ...]:
        """Return a tuple of expressions representing the number of work items
        in each dimension of the kernel's iteration space,
        ordered from fastest to slowest dimension.
        """
        ispace = self._ctx.get_iteration_space()
        match ispace:
            case FullIterationSpace():
                dimensions = ispace.dimensions_in_loop_order()[::-1]

                from ..backend.ast.analysis import collect_undefined_symbols as collect

                for i, dim in enumerate(dimensions):
                    symbs = collect(dim.start) | collect(dim.stop) | collect(dim.step)
                    for ctr in ispace.counters:
                        if ctr in symbs:
                            raise CodegenError(
                                "Unable to construct GPU launch grid constraints for this kernel: "
                                f"Limits in dimension {i} "
                                f"depend on another dimension's counter {ctr.name}"
                            )

                return tuple(ispace.actual_iterations(dim) for dim in dimensions)

            case SparseIterationSpace():
                return (self._ast_factory.parse_index(ispace.index_list.shape[0]),)

            case _:
                assert False, "unexpected iteration space"
