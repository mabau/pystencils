"""
JIT compilation in the ``nbackend`` is managed by subclasses of `JitBase`.
A JIT compiler may freely be created and configured by the user.
It can then be passed to `create_kernel` using the ``jit`` argument of
`CreateKernelConfig`, in which case it is hooked into the `KernelFunction.compile` method
of the generated kernel function::

    my_jit = MyJit()
    kernel = create_kernel(ast, CreateKernelConfig(jit=my_jit))
    func = kernel.compile()

Otherwise, a JIT compiler may also be created free-standing, with the same effect::

    my_jit = MyJit()
    kernel = create_kernel(ast)
    func = my_jit.compile(kernel)

Currently, only wrappers around the legacy JIT compilers are available.

Legacy Just-In-Time Compilation
-------------------------------

Historically, pystencils provides two main pathways for just-in-time compilation:
The ``cpu.cpujit`` module for CPU kernels, and the ``gpu.gpujit`` module for device kernels.
Both are available here through `LegacyCpuJit` and `LegacyGpuJit`.

"""

from .jit import JitBase, NoJit, KernelWrapper
from .legacy_cpu import LegacyCpuJit
from .gpu_cupy import CupyJit

no_jit = NoJit()
"""Disables just-in-time compilation for a kernel."""

__all__ = [
    "JitBase",
    "KernelWrapper",
    "LegacyCpuJit",
    "NoJit",
    "no_jit",
    "CupyJit",
]
