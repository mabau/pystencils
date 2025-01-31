# JIT Compilation

## Base Infrastructure

```{eval-rst}
.. module:: pystencils.jit

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

    KernelWrapper
    JitBase
    NoJit

.. autodata:: no_jit
```

## Legacy CPU JIT

The legacy CPU JIT Compiler is a leftover from pystencils 1.3
which at the moment still drives most CPU JIT-compilation within the package,
until the new JIT compiler is ready to take over.

```{eval-rst}
.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  LegacyCpuJit
```

## CPU Just-In-Time Compiler

:::{note}
The new CPU JIT compiler is still considered experimental and not yet adopted by most of pystencils.
While the APIs described here will (probably) become the default for pystencils 2.0
and can (and should) already be used for testing,
the current implementation is still *very slow*.
For more information, see [issue !120](https://i10git.cs.fau.de/pycodegen/pystencils/-/issues/120).
:::

To configure and create an instance of the CPU JIT compiler, use the `CpuJit.create` factory method:

:::{card}
```{eval-rst}
.. autofunction:: pystencils.jit.CpuJit.create
  :no-index:
```
:::

### Compiler Infos

The CPU JIT compiler invokes a host C++ compiler to compile and link a Python extension
module containing the generated kernel.
The properties of the host compiler are defined in a `CompilerInfo` object.
To select a custom host compiler and customize its options, set up and pass
a custom compiler info object to `CpuJit.create`.

```{eval-rst}
.. module:: pystencils.jit.cpu.compiler_info

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  CompilerInfo
  GccInfo
  ClangInfo
```

### Implementation

```{eval-rst}
.. module:: pystencils.jit.cpu

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  CpuJit
  cpujit.ExtensionModuleBuilderBase
```

## CuPy-based GPU JIT

```{eval-rst}
.. module:: pystencils.jit.gpu_cupy

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  CupyJit
  CupyKernelWrapper
  LaunchGrid
```
