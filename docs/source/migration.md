---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
mystnb:
  execution_mode: cache
---

(_page_v2_migration)=
# Version 2.0 Migration Guide

With version 2.0, many APIs of *pystencils* will be changed; old interfaces are being deprecated
and new systems are put in place.
This page is a still-incomplete list of these changes, with advice on how to migrate your code
from pystencils 1.x to pystencils 2.0.

```{code-cell} ipython3
:tags: [remove-cell]

import pystencils as ps
```


## Kernel Creation

### Configuration

The API of {any}`create_kernel`, and the configuration options of the {any}`CreateKernelConfig`, have changed significantly.
The `CreateKernelConfig` class has been refined to be safe to copy and edit incrementally.
The recommended way of setting up the code generator is now *incremental configuration*:

```{code-cell} ipython3
cfg = ps.CreateKernelConfig()
cfg.default_dtype = "float32"
cfg.cpu.openmp.enable = True
cfg.cpu.openmp.num_threads = 8
cfg.ghost_layers = 2
```

- *Data Types:* `CreateKernelConfig` now takes to parameters to control data types in your kernels:
  the ``default_dtype`` is applied to all numerical computations, while the ``index_dtype`` is used
  for all index calculations and loop counters.
- *CPU Optimization Options:* Should now be set via the {any}`cpu <CpuOptions>` option category and its subcategories.
   
:::{dropdown} Deprecated options of `CreateKernelConfig`

- ``data_type``: Use ``default_dtype`` instead
- ``cpu_openmp``: Set OpenMP-Options in the `cpu.openmp <OpenMpOptions>` category instead.
- ``cpu_vectorize_info``: Set vectorization options in the `cpu.vectorize <VectorizationOptions>` category instead
- ``gpu_indexing_params``: Set GPU indexing options in the `gpu <GpuOptions>` category instead

:::


### Type Checking

The old type checking system of pystencils' code generator has been replaced by a new type inference and validation
mechanism whose rules are much stricter than before.
While running `create_kernel`, you may now encounter a `TypificationError` where previously your kernels compiled fine.
If this happens, it is probable that you have been doing some illegal, maybe dangerous, or at least unsafe things with data types
(like inserting integers into a floating-point context without casting them, or mixing types of different precisions or signedness).
If you are sure the error is not your fault, please file an issue at our
[bug tracker](https://i10git.cs.fau.de/pycodegen/pystencils/-/issues).

### Type System

The ``pystencils.typing`` module has been entirely replaced by the new {any}`pystencils.types` module,
which is home to a completely new type system.
The primary interaction points with this system are still the {any}`TypedSymbol` class and the {any}`create_type` routine.
Code using any of these two should not require any changes, except:

- *Importing `TypedSymbol` and `create_type`:* Both `TypedSymbol` and `create_type` should now be imported directly
  from the ``pystencils`` namespace.
- *Custom data types:* `TypedSymbol` used to accept arbitrary strings as data types.
  This is no longer possible; instead, import {any}`pystencils.types.PsCustomType` and use it to describe
  custom data types unknown to pystencils, as in ``TypedSymbol("xs", PsCustomType("std::vector< int >"))``

All old data type classes (such as ``BasicType``, ``PointerType``, ``StructType``, etc.) have been removed
and replaced by the class hierarchy below {any}`PsType`.
Directly using any of these type classes in the frontend is discouraged unless absolutely necessary;
in most cases, `create_type` suffices.

