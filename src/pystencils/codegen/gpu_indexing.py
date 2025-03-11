from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast, Any, Callable
from itertools import chain

from .functions import Lambda
from .parameters import Parameter
from .errors import CodegenError
from .config import GpuIndexingScheme, _AUTO_TYPE

from ..backend.kernelcreation import (
    KernelCreationContext,
    FullIterationSpace,
    SparseIterationSpace,
)
from ..backend.platforms.cuda import ThreadMapping

from ..backend.ast.expressions import PsExpression


dim3 = tuple[int, int, int]
_Dim3Lambda = tuple[Lambda, Lambda, Lambda]


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


class AutomaticLaunchConfiguration(GpuLaunchConfiguration):
    """Launch configuration that is dynamically computed from kernel parameters.

    This launch configuration permits no further user customization.
    """

    def __init__(
        self,
        block_size: _Dim3Lambda,
        grid_size: _Dim3Lambda,
    ) -> None:
        self._block_size = block_size
        self._grid_size = grid_size

        self._params: frozenset[Parameter] = frozenset().union(
            *(lb.parameters for lb in chain(block_size, grid_size))
        )

    @property
    def parameters(self) -> frozenset[Parameter]:
        return self._params

    def evaluate(self, **kwargs) -> tuple[dim3, dim3]:
        block_size = tuple(int(bs(**kwargs)) for bs in self._block_size)
        grid_size = tuple(int(gs(**kwargs)) for gs in self._grid_size)
        return cast(dim3, block_size), cast(dim3, grid_size)

    def jit_cache_key(self) -> Any:
        return ()


class ManualLaunchConfiguration(GpuLaunchConfiguration):
    """Manual GPU launch configuration.

    This launch configuration requires the user to set block and grid size.
    """

    def __init__(
        self,
    ) -> None:
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

        return self._block_size, self._grid_size

    def jit_cache_key(self) -> Any:
        return (self._block_size, self._grid_size)


class DynamicBlockSizeLaunchConfiguration(GpuLaunchConfiguration):
    """GPU launch configuration that permits the user to set a block size and dynamically computes the grid size.

    The actual launch grid size is computed from the user-defined ``user_block_size`` and the number of work items
    in the kernel's iteration space as follows.
    For each dimension :math:`c \\in \\{ x, y, z \\}`,

    - if ``user_block_size.c > num_work_items.c``, ``block_size = num_work_items.c`` and ``grid_size.c = 1``;
    - otherwise, ``block_size.c = user_block_size.c`` and ``grid_size.c = ceil(num_work_items.c / block_size.c)``.
    """

    def __init__(
        self,
        num_work_items: _Dim3Lambda,
        default_block_size: dim3 | None = None,
    ) -> None:
        self._num_work_items = num_work_items

        self._block_size: dim3 | None = default_block_size

        self._params: frozenset[Parameter] = frozenset().union(
            *(wit.parameters for wit in num_work_items)
        )

    @property
    def num_work_items(self) -> _Dim3Lambda:
        """Lambda expressions that compute the number of work items in each iteration space
        dimension from kernel parameters."""
        return self._num_work_items

    @property
    def block_size(self) -> dim3 | None:
        """The desired GPU block size."""
        return self._block_size

    @block_size.setter
    def block_size(self, val: dim3):
        self._block_size = val

    @property
    def parameters(self) -> frozenset[Parameter]:
        """Parameters of this launch configuration"""
        return self._params

    def evaluate(self, **kwargs) -> tuple[dim3, dim3]:
        if self._block_size is None:
            raise AttributeError("No GPU block size was specified by the user!")

        from ..utils import div_ceil

        num_work_items = cast(
            dim3, tuple(int(wit(**kwargs)) for wit in self._num_work_items)
        )
        reduced_block_size = cast(
            dim3,
            tuple(min(wit, bs) for wit, bs in zip(num_work_items, self._block_size)),
        )
        grid_size = cast(
            dim3,
            tuple(
                div_ceil(wit, bs) for wit, bs in zip(num_work_items, reduced_block_size)
            ),
        )

        return reduced_block_size, grid_size

    def jit_cache_key(self) -> Any:
        return self._block_size


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
        scheme: GpuIndexingScheme,
        default_block_size: dim3 | _AUTO_TYPE | None = None,
        manual_launch_grid: bool = False,
    ) -> None:
        self._ctx = ctx
        self._scheme = scheme
        self._default_block_size = default_block_size
        self._manual_launch_grid = manual_launch_grid

        from ..backend.kernelcreation import AstFactory
        from .driver import KernelFactory

        self._ast_factory = AstFactory(self._ctx)
        self._kernel_factory = KernelFactory(self._ctx)

    def get_thread_mapping(self) -> ThreadMapping:
        """Retrieve a thread mapping object for use by the backend"""

        from ..backend.platforms.cuda import Linear3DMapping, Blockwise4DMapping

        match self._scheme:
            case GpuIndexingScheme.Linear3D:
                return Linear3DMapping()
            case GpuIndexingScheme.Blockwise4D:
                return Blockwise4DMapping()

    def get_launch_config_factory(self) -> Callable[[], GpuLaunchConfiguration]:
        """Retrieve a factory for the launch configuration for later consumption by the runtime system"""
        if self._manual_launch_grid:
            return ManualLaunchConfiguration

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
            self._ast_factory.parse_index(1)
            for _ in range(3 - rank)
        )
        
        num_work_items = cast(
            _Dim3Lambda,
            tuple(self._kernel_factory.create_lambda(wit) for wit in work_items_expr),
        )

        def factory():
            return DynamicBlockSizeLaunchConfiguration(
                num_work_items,
                self._get_default_block_size(rank),
            )

        return factory

    def _get_default_block_size(self, rank: int) -> dim3:
        if self._default_block_size is None:
            raise CodegenError("The default block size option was not set")

        if isinstance(self._default_block_size, _AUTO_TYPE):
            match rank:
                case 1:
                    return (256, 1, 1)
                case 2:
                    return (128, 2, 1)
                case 3:
                    return (128, 2, 2)
                case _:
                    assert False, "unreachable code"
        else:
            return self._default_block_size

    def _get_blockwise4d_config_factory(
        self,
    ) -> Callable[[], AutomaticLaunchConfiguration]:
        work_items = self._get_work_items()
        rank = len(work_items)

        if rank > 4:
            raise ValueError(f"Iteration space rank is too large: {rank}")

        block_size = (
            self._kernel_factory.create_lambda(work_items[0]),
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
