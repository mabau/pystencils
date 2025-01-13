from __future__ import annotations
from typing import Sequence, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..codegen import Kernel, Parameter, Target


class JitError(Exception):
    """Indicates an error during just-in-time compilation"""


class KernelWrapper(ABC):
    """Wrapper around a compiled and executable pystencils kernel."""

    def __init__(self, kfunc: Kernel) -> None:
        self._kfunc = kfunc

    @abstractmethod
    def __call__(self, **kwargs) -> None:
        pass

    @property
    def kernel_function(self) -> Kernel:
        return self._kfunc
    
    @property
    def ast(self) -> Kernel:
        return self._kfunc
    
    @property
    def target(self) -> Target:
        return self._kfunc.target
    
    @property
    def parameters(self) -> Sequence[Parameter]:
        return self._kfunc.parameters

    @property
    def code(self) -> str:
        from pystencils.display_utils import get_code_str

        return get_code_str(self._kfunc)


class JitBase(ABC):
    """Base class for just-in-time compilation interfaces implemented in pystencils."""

    @abstractmethod
    def compile(self, kernel: Kernel) -> KernelWrapper:
        """Compile a kernel function and return a callable object which invokes the kernel."""


class NoJit(JitBase):
    """Not a JIT compiler: Used to explicitly disable JIT compilation on an AST."""

    def compile(self, kernel: Kernel) -> KernelWrapper:
        raise JitError(
            "Just-in-time compilation of this kernel was explicitly disabled."
        )
