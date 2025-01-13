"""
JIT compilation is realized by subclasses of `JitBase`.
A JIT compiler may freely be created and configured by the user.
It can then be passed to `create_kernel` using the ``jit`` argument of
`CreateKernelConfig`, in which case it is hooked into the `Kernel.compile` method
of the generated kernel function::

    my_jit = MyJit()
    kernel = create_kernel(ast, CreateKernelConfig(jit=my_jit))
    func = kernel.compile()

Otherwise, a JIT compiler may also be created free-standing, with the same effect::

    my_jit = MyJit()
    kernel = create_kernel(ast)
    func = my_jit.compile(kernel)

For GPU kernels, a basic JIT-compiler based on cupy is available (`CupyJit`).
For CPU kernels, at the moment there is only `LegacyCpuJit`, which is a wrapper
around the legacy CPU compiler wrapper used by pystencils 1.3.x.
It is due to be replaced in the near future.

"""

from .jit import JitBase, NoJit, KernelWrapper
from .legacy_cpu import LegacyCpuJit
from .gpu_cupy import CupyJit, CupyKernelWrapper, LaunchGrid

no_jit = NoJit()
"""Disables just-in-time compilation for a kernel."""

__all__ = [
    "JitBase",
    "KernelWrapper",
    "LegacyCpuJit",
    "NoJit",
    "no_jit",
    "CupyJit",
    "CupyKernelWrapper",
    "LaunchGrid"
]
