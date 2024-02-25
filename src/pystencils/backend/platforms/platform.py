from abc import ABC, abstractmethod

from ..ast import PsBlock

from ..kernelcreation.context import KernelCreationContext
from ..kernelcreation.iteration_space import IterationSpace


class Platform(ABC):
    """Abstract base class for all supported platforms.

    The platform performs all target-dependent tasks during code generation:

     - Translation of the iteration space to an index source (loop nest, GPU indexing, ...)
     - Platform-specific optimizations (e.g. vectorization, OpenMP)
    """

    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx

    @property
    @abstractmethod
    def required_headers(self) -> set[str]:
        pass

    @abstractmethod
    def materialize_iteration_space(
        self, block: PsBlock, ispace: IterationSpace
    ) -> PsBlock:
        ...

    @abstractmethod
    def optimize(self, kernel: PsBlock) -> PsBlock:
        ...
