# GPU Code Generation

The code generation infrastructure for Nvidia and AMD GPUs using CUDA and HIP comprises the following components:

 - The {any}`CudaPlatform` at `backend.platforms` which performs materialization of a kernel's iteration
   space by mapping GPU block and thread indices to iteration space points. To perform this task,
   it depends on a {any}`ThreadMapping` instance which defines the nature of that mapping.
   The platform also takes care of lowering mathematical functions to their CUDA runtime library implementation.
 - In the code generation driver, the strings are drawn by the `GpuIndexing` helper class.
   It provides both the {any}`ThreadMapping` for the codegen backend, as well as the launch configuration
   for the runtime system.

:::{attention}

Code generation for HIP through the `CudaPlatform` is experimental and not tested at the moment.
:::

## The CUDA Platform and Thread Mappings

```{eval-rst}
.. module:: pystencils.backend.platforms.cuda

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    ThreadMapping
    Linear3DMapping
    Blockwise4DMapping
```

## Thread Indexing In The Driver

With regard to GPU thread indexing, the code generation driver has two tasks:
it must provide the Cuda platform object with a valid thread mapping,
and must also provide the runtime system with a [launch configuration](#gpu_launch_config)
which defines the shape of the GPU block grid.
Both of these are produced by the {any}`GpuIndexing` class.
It is instantiated with the GPU indexing scheme and indexing options given by the user.

At this time, the backend and code generation driver support two indexing schemes:
"Linear3D" (see {any}`Linear3DMapping`) and "Blockwise4D" (see {any}`Blockwise4DMapping`).
These are mostly reimplemented from the pystencils 1.3.x `"block"` and `"line"` indexing options.
The GPU indexing system may be extended in the future.


```{eval-rst}
.. module:: pystencils.codegen.gpu_indexing

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    GpuIndexing
```

(gpu_launch_config)=
## The Launch Configuration

The launch configuration is attached to the `GpuKernel` and thus returned to the runtime system.
Since a concrete launch configuration is not specific to the kernel itself, but to the kernels'
invocation site, the code generator only attaches a *factory function* for launch configurations
to `GpuKernel`. It is up to the runtime system to locally instantiate and configure a launch configuration.
To determine the actual launch grid, the launch configuration must be evaluated at the kernel's call site
by passing the required parameters to `GpuLaunchConfiguration.evaluate`

The {any}`CupyJit`, for instance, will create the launch configuration object while preparing the JIT-compiled
kernel wrapper object. The launch config is there exposed to the user, who may modify some of its properties.
These depend on the type of the launch configuration:
while the `AutomaticLaunchConfiguration` permits no modification and computes grid and block size directly from kernel
parameters,
the `ManualLaunchConfiguration` requires the user to manually specifiy both grid and block size.
The `DynamicBlockSizeLaunchConfiguration` dynamically computes the grid size from either the default block size
or a computed block size. Computing block sizes can be signaled by the user via the `trim_block_size` or 
`fit_block_size` member functions. These function receive an initial block size as an argument and adapt it.
The `trim_block_size` function trims the initial block size with the sizes of the iteration space, i.e. it takes 
the minimum value of both sizes per dimension. The `fit_block_size` performs a block fitting algorithm that adapts 
the initial block size by incrementally enlarging the trimmed block size until it is large enough 
and aligns with the warp size.

The `evaluate` method can only be used from within a Python runtime environment.
When exporting pystencils CUDA kernels for external use in C++ projects,
equivalent C++ code evaluating the launch config must be generated.
This is the task of, e.g., [pystencils-sfg](https://pycodegen.pages.i10git.cs.fau.de/pystencils-sfg/).


```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/entire_class.rst

    GpuLaunchConfiguration
    AutomaticLaunchConfiguration
    ManualLaunchConfiguration
    DynamicBlockSizeLaunchConfiguration
```
