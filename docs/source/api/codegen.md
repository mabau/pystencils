# Code Generation

## Invocation

```{eval-rst}
.. module:: pystencils.codegen

.. autosummary::
  :toctree: generated
  :nosignatures:

  create_kernel
```
  
## Configuration

```{eval-rst}
.. module:: pystencils.codegen.config
```

The code generation driver (`create_kernel`, but also `DefaultKernelCreationDriver`) can be configured by
passing it a `CreateKernelConfig` object.
This object can be constructed incrementally:

```Python
cfg = ps.CreateKernelConfig()
cfg.default_dtype = "float32"
cfg.target = ps.Target.X86_AVX
cfg.cpu.openmp.enable = True
cfg.cpu.vectorize.enable = True
cfg.cpu.vectorize.assume_inner_stride_one = True
```

### Options and Option Categories

The following options and option categories are exposed by the configuration object:

#### Target Specification

```{eval-rst}
.. current

.. autosummary::

  ~CreateKernelConfig.target
```

#### Data Types

```{eval-rst}
.. autosummary::

  ~CreateKernelConfig.default_dtype
  ~CreateKernelConfig.index_dtype
```

#### Iteration Space

```{eval-rst}
.. autosummary::

  ~CreateKernelConfig.ghost_layers
  ~CreateKernelConfig.iteration_slice
  ~CreateKernelConfig.index_field
```

#### Kernel Constraint Checks

```{eval-rst}
.. autosummary::

  ~CreateKernelConfig.allow_double_writes
  ~CreateKernelConfig.skip_independence_check
```

#### Target-Specific Options

The following categories with target-specific options are exposed:

| | |
|---------------------------|--------------------------|
| {any}`cpu <CpuOptions>`   | Options for CPU kernels  |
| {any}`gpu <GpuOptions>`   | Options for GPU kernels  |
| {any}`sycl <SyclOptions>` | Options for SYCL kernels |


#### Kernel Object and Just-In-Time Compilation

```{eval-rst}
.. autosummary::

  ~CreateKernelConfig.function_name
  ~CreateKernelConfig.jit
```

### Configuration System Classes

```{eval-rst}

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/recursive_class.rst

  CreateKernelConfig
  CpuOptions
  OpenMpOptions
  VectorizationOptions
  GpuOptions
  SyclOptions
  GpuIndexingScheme

.. autosummary::
  :toctree: generated
  :nosignatures:

  AUTO

.. dropdown:: Implementation Details

  .. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    Option
    BasicOption
    Category
    ConfigBase

```

## Target Specification

```{eval-rst}

.. module:: pystencils.codegen.target

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/recursive_class.rst

  Target

```

## Code Generation Drivers

```{eval-rst}
.. module:: pystencils.codegen.driver

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  DefaultKernelCreationDriver

.. autosummary::
  :toctree: generated
  :nosignatures:

  get_driver
```

## Output Code Objects

```{eval-rst}
.. currentmodule:: pystencils.codegen

.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/entire_class.rst

  Kernel
  GpuKernel
  Parameter
  Lambda
```
