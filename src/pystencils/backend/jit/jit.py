from __future__ import annotations
from typing import Callable, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..kernelfunction import KernelFunction


class JitError(Exception):
    """Indicates an error during just-in-time compilation"""


class JitBase(ABC):
    """Base class for just-in-time compilation interfaces implemented in pystencils."""

    @abstractmethod
    def compile(self, kernel: KernelFunction) -> Callable[..., None]:
        """Compile a kernel function and return a callable object which invokes the kernel."""


class NoJit(JitBase):
    """Not a JIT compiler: Used to explicitly disable JIT compilation on an AST."""

    def compile(self, kernel: KernelFunction) -> Callable[..., None]:
        raise JitError(
            "Just-in-time compilation of this kernel was explicitly disabled."
        )


class LegacyCpuJit(JitBase):
    """Wrapper around ``pystencils.cpu.cpujit``"""

    def compile(self, kernel: KernelFunction) -> Callable[..., None]:
        from .legacy_cpu import compile_and_load

        return compile_and_load(kernel)


class LegacyGpuJit(JitBase):
    """Wrapper around ``pystencils.gpu.gpujit``"""

    def compile(self, kernel: KernelFunction) -> Callable[..., None]:
        from ...old.gpu.gpujit import make_python_function

        return make_python_function(kernel)
