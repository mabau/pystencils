from __future__ import annotations
from typing import Sequence
from abc import abstractmethod

from ..ast.expressions import PsExpression
from ..ast.structural import PsBlock
from ..kernelcreation.iteration_space import (
    IterationSpace,
    FullIterationSpace,
    SparseIterationSpace,
)
from .platform import Platform


class GpuThreadsRange:
    """Number of threads required by a GPU kernel, in order (x, y, z)."""

    @staticmethod
    def from_ispace(ispace: IterationSpace) -> GpuThreadsRange:
        if isinstance(ispace, FullIterationSpace):
            return GpuThreadsRange._from_full_ispace(ispace)
        elif isinstance(ispace, SparseIterationSpace):
            work_items = (PsExpression.make(ispace.index_list.shape[0]),)
            return GpuThreadsRange(work_items)
        else:
            assert False

    def __init__(
        self,
        num_work_items: Sequence[PsExpression],
    ):
        self._dim = len(num_work_items)
        self._num_work_items = tuple(num_work_items)

    # @property
    # def grid_size(self) -> tuple[PsExpression, ...]:
    #     return self._grid_size

    # @property
    # def block_size(self) -> tuple[PsExpression, ...]:
    #     return self._block_size

    @property
    def num_work_items(self) -> tuple[PsExpression, ...]:
        """Number of work items in (x, y, z)-order."""
        return self._num_work_items

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _from_full_ispace(ispace: FullIterationSpace) -> GpuThreadsRange:
        dimensions = ispace.dimensions_in_loop_order()[::-1]
        if len(dimensions) > 3:
            raise NotImplementedError(
                f"Cannot create a GPU threads range for an {len(dimensions)}-dimensional iteration space"
            )
        work_items = [ispace.actual_iterations(dim) for dim in dimensions]
        return GpuThreadsRange(work_items)


class GenericGpu(Platform):
    @abstractmethod
    def materialize_iteration_space(
        self, block: PsBlock, ispace: IterationSpace
    ) -> tuple[PsBlock, GpuThreadsRange]:
        pass
