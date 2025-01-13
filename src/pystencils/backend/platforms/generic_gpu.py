from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod

from ..ast.expressions import PsExpression
from ..ast.structural import PsBlock
from ..kernelcreation.iteration_space import (
    IterationSpace,
    FullIterationSpace,
    SparseIterationSpace,
)
from .platform import Platform
from ..exceptions import MaterializationError

if TYPE_CHECKING:
    from ...codegen.kernel import GpuThreadsRange


class GenericGpu(Platform):
    @abstractmethod
    def materialize_iteration_space(
        self, body: PsBlock, ispace: IterationSpace
    ) -> tuple[PsBlock, GpuThreadsRange | None]:
        pass

    @classmethod
    def threads_from_ispace(cls, ispace: IterationSpace) -> GpuThreadsRange:
        from ...codegen.kernel import GpuThreadsRange

        if isinstance(ispace, FullIterationSpace):
            return cls._threads_from_full_ispace(ispace)
        elif isinstance(ispace, SparseIterationSpace):
            work_items = (PsExpression.make(ispace.index_list.shape[0]),)
            return GpuThreadsRange(work_items)
        else:
            assert False

    @classmethod
    def _threads_from_full_ispace(cls, ispace: FullIterationSpace) -> GpuThreadsRange:
        from ...codegen.kernel import GpuThreadsRange
        
        dimensions = ispace.dimensions_in_loop_order()[::-1]
        if len(dimensions) > 3:
            raise NotImplementedError(
                f"Cannot create a GPU threads range for an {len(dimensions)}-dimensional iteration space"
            )
        
        from ..ast.analysis import collect_undefined_symbols as collect

        for dim in dimensions:
            symbs = collect(dim.start) | collect(dim.stop) | collect(dim.step)
            for ctr in ispace.counters:
                if ctr in symbs:
                    raise MaterializationError(
                        "Unable to construct GPU threads range for iteration space: "
                        f"Limits of dimension counter {dim.counter.name} "
                        f"depend on another dimension's counter {ctr.name}"
                    )

        work_items = [ispace.actual_iterations(dim) for dim in dimensions]
        return GpuThreadsRange(work_items)
