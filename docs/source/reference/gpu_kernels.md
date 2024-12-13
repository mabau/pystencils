---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
mystnb:
  execution_mode: cache
---

```{code-cell} ipython3
:tags: [remove-cell]

import sympy as sp
import pystencils as ps
import numpy as np
import matplotlib.pyplot as plt
```

(guide_gpukernels)=
# Pystencils for GPUs

Pystencils offers code generation for Nvidia GPUs using the CUDA programming model,
as well as just-in-time compilation and execution of CUDA kernels from within Python
based on the [cupy] library.w
This section's objective is to give a detailed introduction into the creation of
GPU kernels with pystencils.

## Generate, Compile and Run CUDA Kernels

In order to obtain a CUDA implementation of a symbolic kernel, naught more is required
than setting the {any}`target <CreateKernelConfig.target>` code generator option to
{any}`Target.CUDA`:

```{code-cell} ipython3
f, g = ps.fields("f, g: float64[3D]")
update = ps.Assignment(f.center(), 2 * g.center())

cfg = ps.CreateKernelConfig(target=ps.Target.CUDA)
kernel = ps.create_kernel(update, cfg)

ps.inspect(kernel)
```

The `kernel` object returned by the code generator in above snippet is an instance
of the {py:class}`GpuKernelFunction` class.
It extends {py:class}`KernelFunction` with some GPU-specific information.
In particular, it defines the {any}`threads_range <GpuKernelFunction.threads_range>`
property, which tells us how many threads the kernel is expecting to be executed with:

```{code-cell} ipython3
kernel.threads_range
```

If a GPU is available and [CuPy][cupy] is installed in the current environment,
the kernel can be compiled and run immediately.
To execute the kernel, a {any}`cupy.ndarray` has to be passed for each field.

:::{note}
[CuPy][cupy] is a Python library for numerical computations on GPU arrays,
which operates much in the same way that [NumPy][numpy] works on CPU arrays.
Cupy and NumPy expose nearly the same APIs for array operations;
the difference being that CuPy allocates all its arrays on the GPU
and performs its operations as CUDA kernels.
Also, CuPy exposes a just-in-time-compiler for GPU kernels, which internally calls [nvcc].
In pystencils, we use CuPy both to compile and provide executable kernels on-demand from within Python code,
and to allocate and manage the data these kernels can be executed on.

For more information on CuPy, refer to [their documentation][cupy-docs].
:::

```{code-cell} ipython3
:tags: [raises-exception]
import cupy as cp

rng = cp.random.default_rng(seed=42)
f_arr = rng.random((16, 16, 16))
g_arr = cp.zeros_like(f_arr)

kfunc = kernel.compile()
kfunc(f=f_arr, g=g_arr)
```

### Modifying the Launch Grid

The `kernel.compile()` invocation in the above code produces a {any}`CupyKernelWrapper` callable object.
This object holds the kernel's launch grid configuration
(i.e. the number of thread blocks, and the number of threads per block.)
Pystencils specifies a default value for the block size and if possible, 
the number of blocks is automatically inferred in order to cover the entire iteration space.
In addition, the wrapper's interface allows us to customize the GPU launch grid,
by manually setting both the number of threads per block, and the number of blocks on the grid:

```{code-cell} ipython3
kfunc.block_size = (16, 8, 8)
kfunc.num_blocks = (1, 2, 2)
```

For most kernels, setting only the `block_size` is sufficient since pystencils will
automatically compute the number of blocks;
for exceptions to this, see [](#manual_launch_grids).
If `num_blocks` is set manually and the launch grid thus specified is too small, only
a part of the iteration space will be traversed by the kernel;
similarily, if it is too large, it will cause any threads working outside of the iteration bounds to idle.

(manual_launch_grids)=
### Manual Launch Grids and Non-Cuboid Iteration Patterns

In some cases, it will be unavoidable to set the launch grid size manually;
especially if the code generator is unable to automatically determine the size of the
iteration space.
An example for this is the triangular iteration previously described in the [Kernel Creation Guide](#example_triangular_iteration).
Let's set it up once more:

```{code-cell} ipython3
:tags: [remove-cell]

def _draw_ispace(f_arr):
    n, m = f_arr.shape
    fig, ax = plt.subplots()
    
    ax.set_xticks(np.arange(0, m, 4))
    ax.set_yticks(np.arange(0, n, 4))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

    ax.set_xticks(np.arange(-.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    
    ax.grid(which="minor", linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    ax.imshow(f_arr, interpolation="none", aspect="equal", origin="lower")
```

```{code-cell} ipython3
:tags: [remove-cell]

f = ps.fields("f: float64[2D]")
assignments = [
    ps.Assignment(f(0), 1)
]
```

```{code-cell} ipython3
y = ps.DEFAULTS.spatial_counters[0]
cfg = ps.CreateKernelConfig(
    target=ps.Target.CUDA,
    iteration_slice=ps.make_slice[:, y:]
)
    
kernel = ps.create_kernel(assignments, cfg).compile()
```

This warns us that the threads range could not be determined automatically.
We can disable this warning by setting `manual_launch_grid` in the GPU indexing options:

```{code-cell}
cfg = ps.CreateKernelConfig(
    # ... other options ...
    gpu_indexing=ps.GpuIndexingConfig(
        manual_launch_grid=True
    )
)
```

Now, to execute our kernel, we have to manually specify its launch grid:

```{code-cell} ipython3
kernel.block_size = (8, 8)
kernel.num_blocks = (2, 2)
```

This way the kernel will cover this iteration space:

```{code-cell} ipython3
:tags: [remove-input]
f_arr = cp.zeros((16, 16))
kernel(f=f_arr)
_draw_ispace(cp.asnumpy(f_arr))
```

We can also observe the effect of decreasing the launch grid size:

```{code-cell} ipython3
kernel.block_size = (4, 4)
kernel.num_blocks = (2, 3)
```

```{code-cell} ipython3
:tags: [remove-input]
f_arr = cp.zeros((16, 16))
kernel(f=f_arr)
_draw_ispace(cp.asnumpy(f_arr))
```

Here, since there are only eight threads operating in $x$-direction, 
and twelve threads in $y$-direction,
only a part of the triangle is being processed.

## API Reference

```{eval-rst}
.. autosummary::
  :toctree: generated
  :nosignatures:
  :template: autosummary/recursive_class.rst

  pystencils.backend.kernelfunction.GpuKernelFunction
  pystencils.backend.jit.gpu_cupy.CupyKernelWrapper
```

:::{admonition} Developers To Do:

- Fast approximation functions
- Fp16 on GPU
:::


[cupy]: https://cupy.dev "CuPy Homepage"
[numpy]: https://numpy.org "NumPy Homepage"
[nvcc]: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html "NVIDIA CUDA Compiler Driver"
[cupy-docs]: https://docs.cupy.dev/en/stable/overview.html "CuPy Documentation"