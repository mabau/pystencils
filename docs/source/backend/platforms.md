# Platforms

All target-specific code generation in the pystencils backend is facilitated
through the *platform classes*.
This includes:

 - Materialization of the iteration space, meaning the mapping of iteration space points to some indexing structure
 - Lowering of mathematical functions to their implementation in some runtime environment
 - Selection of vector intrinsics for SIMD-capable CPU targets

Encapsulation of hardware- and environment-specific details into platform objects allows
us to implement most of the code generator in a generic and hardware-agnostic way.
It also makes it easier to extend pystencils with support for additional code generation
targets in the future.

## Base Classes

```{eval-rst}
.. module:: pystencils.backend.platforms

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    Platform
    GenericCpu
    GenericVectorCpu
    GenericGpu
```

## CPU Platforms

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    X86VectorCpu
    X86VectorArch
```

## GPU Platforms

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    CudaPlatform
    SyclPlatform
```
