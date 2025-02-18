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
based on the [cupy] library.
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
of the {py:class}`GpuKernel` class.
It extends {py:class}`Kernel` with some GPU-specific information.

If a GPU is available and [CuPy][cupy] is installed in the current environment,
the kernel can be compiled and run immediately.
To execute the kernel, a {any}`cupy.ndarray` has to be passed for each field:

```{code-cell} ipython3
:tags: [raises-exception]
import cupy as cp

rng = cp.random.default_rng(seed=42)
f_arr = rng.random((16, 16, 16))
g_arr = cp.zeros_like(f_arr)

kfunc = kernel.compile()
kfunc(f=f_arr, g=g_arr)
```

:::{note}
[CuPy][cupy] is a Python library for numerical computations on GPU arrays,
which operates much in the same way that [NumPy][numpy] works on CPU arrays.
Cupy and NumPy expose nearly the same APIs for array operations;
the difference being that CuPy allocates all its arrays on the GPU
and performs its operations as CUDA kernels.
Also, CuPy exposes a just-in-time-compiler for GPU kernels, which internally calls [nvrtc].
In pystencils, we use CuPy both to compile and provide executable kernels on-demand from within Python code,
and to allocate and manage the data these kernels can be executed on.

For more information on CuPy, refer to [their documentation][cupy-docs].
:::

(indexing_and_launch_config)=
## Modify the Indexing Scheme and Launch Configuration

There are two key elements to how the work items of a GPU kernel's iteration space
are mapped onto a GPU launch grid:
 - The *indexing scheme* defines the relation between thread indices and iteration space points;
   it can be modified through the {any}`gpu.indexing_scheme <GpuOptions.indexing_scheme>` option
   and is fixed for the entire kernel.
 - The *launch configuration* defines the number of threads per block, and the number of blocks on the grid,
   with which the kernel should be launched.
   The launch configuration mostly depends on the size of the arrays passed to the kernel,
   but parts of it may also be modified.
   The launch configuration may change at each kernel invocation.

(linear3d)=
### The Default "Linear3D" Indexing Scheme

By default, *pystencils* will employ a 1:1-mapping between threads and iteration space points
via the global thread indices inside the launch grid; e.g.

```{code-block} C++
ctr_0 = start_0 + step_0 * (blockSize.x * blockIdx.x + threadIdx.x);
ctr_1 = start_1 + step_1 * (blockSize.y * blockIdx.y + threadIdx.y);
ctr_2 = start_2 + step_2 * (blockSize.z * blockIdx.z + threadIdx.z);
```

For most kernels with an at most three-dimensional iteration space,
this behavior is sufficient and desired.
It can be enforced by setting `gpu.indexing_scheme = "Linear3D"`.

If the `Linear3D` indexing scheme is used, you may modifiy the GPU thread block size in two places.
The default block size for the kernel can be set via the {any}`gpu.block_size <GpuOptions.block_size>` 
code generator option;
if none is specified, a default depending on the iteration space's dimensionality will be used.

The block size can furthermore be modified at the compiled kernel's wrapper object via the
`launch_config.block_size` attribute:

```{code-cell} ipython3
kfunc = kernel.compile()
kfunc.launch_config.block_size = (256, 2, 1)

# Run the kernel
kfunc(f=f_arr, g=g_arr)
```

In any case. pystencils will automatically compute the grid size from the shapes of the kernel's array arguments
and the given thread block size.

:::{attention}

According to the way GPU architecture splits thread blocks into warps,
pystencils will map the kernel's *fastest* spatial coordinate onto the `x` block and thread
indices, the second-fastest to `y`, and the slowest coordiante to `z`.

This can mean that, when using `cupy` arrays with the default memory layout
(corresponding to the `"numpy"` field layout specifier),
the *thread coordinates* and the *spatial coordinates*
map to each other in *opposite order*; e.g.

| Spatial Coordinate | Thread Index  |
|--------------------|---------------|
| `x` (slowest)      | `threadIdx.z` |
| `y`                | `threadIdx.y` |
| `z` (fastest)      | `threadIdx.x` |

:::

(manual_launch_grids)=
### Manual Launch Grids and Non-Cuboid Iteration Patterns

By default, the above indexing schemes will automatically compute the GPU launch configuration
from array shapes and optional user input.
However, it is also possible to override this behavior and instead specify a launch grid manually.
This will even be unavoidable if the code generator cannot precompute the number of points
in the kernel's iteration space.

To specify a manual launch configuration, set the {any}`gpu.manual_launch_grid <GpuOptions.manual_launch_grid>`
option to `True`.
Then, after compiling the kernel, set its block and grid size via the `launch_config` property:

```{code-cell} ipython3
cfg.gpu.manual_launch_grid = True

kernel = ps.create_kernel(update, cfg)
kfunc = kernel.compile()
kfunc.launch_config.block_size = (64, 2, 1)
kfunc.launch_config.grid_size = (4, 2, 1)
```

An example where this is necessary is the triangular iteration
previously described in the [Kernel Creation Guide](#example_triangular_iteration).
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
cfg = ps.CreateKernelConfig()
cfg.target= ps.Target.CUDA
cfg.iteration_slice = ps.make_slice[:, y:]
```

For this kernel, the code generator cannot figure out a launch configuration on its own,
so we need to manually provide one:

```{code-cell} ipython3
cfg.gpu.manual_launch_grid = True
    
kernel = ps.create_kernel(assignments, cfg).compile()

kernel.launch_config.block_size = (8, 8)
kernel.launch_config.grid_size = (2, 2)
```

This way the kernel will cover this iteration space:

```{code-cell} ipython3
:tags: [remove-input]
f_arr = cp.zeros((16, 16))
kernel(f=f_arr)
_draw_ispace(cp.asnumpy(f_arr))
```

We can also observe the effect of decreasing the launch grid size.

```{code-cell} ipython3
kernel.launch_config.block_size = (4, 4)
kernel.launch_config.grid_size = (2, 3)
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


:::{admonition} Developers To Do:

- Fast approximation functions
- Fp16 on GPU
:::


[cupy]: https://cupy.dev "CuPy Homepage"
[numpy]: https://numpy.org "NumPy Homepage"
[nvrtc]: https://docs.nvidia.com/cuda/nvrtc/index.html "NVIDIA Runtime Compilation Library"
[cupy-docs]: https://docs.cupy.dev/en/stable/overview.html "CuPy Documentation"
