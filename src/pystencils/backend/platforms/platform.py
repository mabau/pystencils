from abc import ABC, abstractmethod

from ..ast.structural import PsBlock
from ..ast.expressions import PsCall, PsExpression

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
        """Set of header files that must be included at the point of definition of a kernel
        running on this platform."""
        pass

    @abstractmethod
    def materialize_iteration_space(
        self, body: PsBlock, ispace: IterationSpace
    ) -> PsBlock:
        """Materialize the given iteration space as an indexing structure and embed the given
        kernel body into that structure."""
        pass

    @abstractmethod
    def select_function(
        self, call: PsCall
    ) -> PsExpression:
        """Select an implementation for the given function on the given data type.

        If no viable implementation exists, raise a `MaterializationError`.
        """
        pass
