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

(guide_kernelcreation)=
# Kernel Creation

Once a kernel's assignments are fully assembled, they need to be passed through pystencils' code
generation engine in order to produce the kernel's executable code.
The goal of this chapter is to shed light on pystencils' main code generation pipeline.
Here, we show how to invoke the code generator and discuss its various configuration options
and their effects on the generated kernel.

## Running the Code Generator

The primary way to invoke the code generation engine is through the {any}`create_kernel` function.
It takes two arguments:
- the list of assignment that make up the kernel (optionally wrapped as an ``AssignmentCollection``),
- and a configuration object, an instance of {any}`CreateKernelConfig <pystencils.codegen.config.CreateKernelConfig>`.

```{eval-rst}
.. currentmodule:: pystencils.codegen

.. autosummary::
  :nosignatures:

  create_kernel
  CreateKernelConfig
```

For a simple kernel, an invocation of the code generator might look like this:

```{code-cell} ipython3
# Symbol and field definitions
u_src, u_dst, f = ps.fields("u_src, u_dst, f: float32[2D]")
h = sp.Symbol("h")

# Kernel definition
update = [
  ps.Assignment(
    u_dst[0,0], (h**2 * f[0, 0] + u_src[1, 0] + u_src[-1, 0] + u_src[0, 1] + u_src[0, -1]) / 4
  )
]

# Code Generator Configuration
cfg = ps.CreateKernelConfig(
  target=ps.Target.CUDA,
  default_dtype="float32",
  ghost_layers=1
)

kernel = ps.create_kernel(update, cfg)
```

The above snippet defines a five-point-stencil Jacobi update. A few noteworthy things are going on:
- The target data type of the kernel is to be `float32`.
  This is explicitly specified for the three fields `u`, `u_tmp` and `f`.
  For the symbol `h`, this is left implicit; `h` therefore learns its data type from the `default_dtype` configuration option.
- The target hardware for this kernel are Nvidia GPUs; this is reflected by the `target` property being set to `Target.CUDA`.
- As the five-point stencil reads data from neighbors offset by one cell, it can not be legally executed on the outermost
  layer of nodes of the fields' 2D arrays. Here, we ensure that these outer layers are excluded by setting `ghost_layers=1`.
  This is not strictly necessary, since the code generator could infer that information by itself.

## Inspecting the Generated Code

The object returned by the code generator, here named `kernel`, is an instance of the {any}`Kernel` class.
This object stores the kernel's name, its list of parameters, the set of fields it operates on, and its hardware target.
Also, it of course holds the kernel itself, in the form of an [abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree) (AST).
This tree can be printed out as compilable code in the target language (C++ or, in this case, CUDA),
but holds much more information than the printed-out code string.
When working in a Jupyter notebook, we can interactively inspect the kernel using pystencils' `inspect` function.
This reveals a widget that allows us investigate various details about the kernel:
- its general properties, such as name, parameters, fields, target, ...;
- its code, represented in the *pystencils IR syntax*;
- the same code, in native C++/CUDA syntax;
- and a visual representation of its abstract syntax tree.

```{code-cell} ipython3
ps.inspect(kernel)
```

## Configuring the Code Generator

The code generation engine can be configured using a wide range of options.
This section aims at explaining the majority of these options,
their interaction and effects, use cases and caveats.


### Target Specification

Pystencils supports code generation for a variety of CPU and GPU hardware.

```{eval-rst}
.. currentmodule:: pystencils.codegen

.. autosummary::
  :nosignatures:

  CreateKernelConfig.target
  Target

```

### Data Types

To produce valid output code, the code generator has to figure out the data types of each
symbol, expression, and assignment occuring inside a kernel.
This happens roughly according to the following rules:
 - **Field Accesses:** Each field has a fixed data type set at its creation, which is also applied to
   each access to that field.
 - **Symbols:** Symbols obtain their data types from two sources. 
   A symbol occuring first on the left-hand side of an assignment receives the data type that
   was inferred for the right-hand side expression of that assignment.
   Symbols occuring first inside some expression on the right-hand side of an assignment, on the other
   hand, receive the {any}`default_dtype <CreateKernelConfig.default_dtype>` set in the {any}`CreateKernelConfig`.

We can observe this behavior by setting up a kernel including several fields with different data types:

```{code-cell} ipython3
from pystencils.sympyextensions import CastFunc

f = ps.fields("f: float32[2D]")
g = ps.fields("g: float16[2D]")

x, y, z = sp.symbols("x, y, z")

assignments = [
  ps.Assignment(x, 42),
  ps.Assignment(y, f(0) + x),
  ps.Assignment(z, g(0))
]

cfg = ps.CreateKernelConfig(
  default_dtype="float32",
  index_dtype="int32"
)

kernel = ps.create_kernel(assignments, cfg)
```

We can take a look at the result produced by the code generator after parsing the above kernel.
Inspecting the internal representation of the kernel's body and loop nest,
we see that `x` has received the `float32` type,
which was specified via the {py:data}`default_dtype <CreateKernelConfig.default_dtype>` option.
The symbol `y`, on the other hand, has inherited its data type `float16` from the access to the field `g`
on its declaration's right-hand side.
Also, we can observe that the loop counters and symbols related to the field's memory layout
are using the `int32` data type, as specified in {py:data}`index_dtype <CreateKernelConfig.index_dtype>`:

```{code-cell} ipython3
:tags: [remove-input]

driver = ps.codegen.get_driver(cfg, retain_intermediates=True)
kernel = driver(assignments)
ps.inspect(driver.intermediates.materialized_ispace, show_cpp=False)
```

:::{note}
To learn more about inspecting code after different stages of the code generator, refer to [this section](#section_codegen_stages).
:::

```{eval-rst}
.. currentmodule:: pystencils.codegen

.. autosummary::
  :nosignatures:

  CreateKernelConfig.default_dtype
  CreateKernelConfig.index_dtype
```

### The Iteration Space

The *domain fields* a kernel operates on are understood to reside on a common,
one-, two- or three-dimensional computational grid.
The grid points may be understood as vertices or cells, depending on the application.
When executed, the kernel performs a computation and updates values on all, or a specific subset
of, these grid points.
The set of points the kernel actually operates on is defined by its *iteration space*.

There are three distinct options to control the iteration space in the code generator,
only one of which can be specified at a time:
 - The ``ghost_layers`` option allows to specify a number of layers of grid points on all
   domain borders that should be excluded from iteration;
 - The ``iteration_slice`` option allows to describe the iteration space using Pythonic slice objects;
 - The ``index_field`` option can be used to realize a sparse list-based iteration by passing a special
   *index field* which holds a list of all points that should be processed.

:::{note}
  The points within a kernel's iteration space are understood to be processed concurrently and in
  no particular order;
  the output of any kernel that relies on some specific iteration order is therefore undefined.
  (When running on a GPU, all grid points might in fact be processed in perfect simultaniety!)
:::

```{eval-rst}
.. currentmodule:: pystencils.codegen

.. autosummary::
  :nosignatures:

  CreateKernelConfig.ghost_layers
  CreateKernelConfig.iteration_slice
  CreateKernelConfig.index_field
```

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

#### Specifying Ghost Layers

One way to alter the iteration space is by introducing ghost layers on the domain borders.
These layers of grid points are stripped from the iterations, and can be used to hold
boundary values or exchange data in MPI-parallel simulations.

##### Automatic Ghost Layers

The easiest way to define an iteration space with ghost layers
is to set `ghost_layers=ps.AUTO`, which is also the default
when no iteration space options are specified.
In this case, the code generator will examine the kernel to find the maximum range
of its stencil -- that is, the maximum neighbor offset encountered in any field access.
If, for instance, a neighbor node in $x$-direction with offset $k$ is accessed by the kernel,
it cannot legally execute on the outermost $k$ layers of nodes in that direction since it would
access memory out-of-bounds.
Therefore, an automatic number of $k$ ghost layers at each domain border is inferred.
As we can see in the example below, the number of inferred ghost layers at each domain border will be set to the maximum required in any dimension.

```{code-cell} ipython3
:tags: [remove-cell]

u, v = ps.fields("u, v: [2D]")
```

To illustrate, the following kernel accesses neighbor nodes with a maximum offset of two:

```{code-cell} ipython3
ranged_update = ps.Assignment(u.center(), v[-2, -1] + v[2, 1])

cfg = ps.CreateKernelConfig(ghost_layers=ps.AUTO)
kernel = ps.create_kernel(ranged_update, cfg)
```

With `ghost_layers=ps.AUTO`, its iteration space will look like this (yellow cells are included, purple cells excluded).

```{code-cell} ipython3
:tags: [remove-input]

f = ps.fields("f: float64[2D]")
assignments = [
    ranged_update,
    ps.Assignment(f(0), 1)
]
kernel = ps.create_kernel(assignments).compile()

f_arr = np.zeros((16, 16))
u_arr = np.zeros_like(f_arr)
v_arr = np.zeros_like(f_arr)

kernel(f=f_arr, u=u_arr, v=v_arr)

_draw_ispace(f_arr)
```

##### Uniform and Nonuniform Ghost Layers

```{code-cell} ipython3
:tags: [remove-cell]

def _show_ispace(cfg):
    f = ps.fields("f: float64[2D]")
    assignments = [
        ps.Assignment(f(0), 1)
    ]
    kernel = ps.create_kernel(assignments, cfg).compile()

    f_arr = np.zeros((16, 16))
    kernel(f=f_arr)

    _draw_ispace(f_arr)
```

Setting `ghost_layers` to a number will remove that many layers from the iteration space in each dimension:

```{code-cell} ipython3
cfg = ps.CreateKernelConfig(
    ghost_layers=1
)
```

```{code-cell} ipython3
:tags: [remove-input]

_show_ispace(cfg)
```

Ghost layers can also be specified individually for each dimension and lower/upper borders,
by passing a sequence with either a single integer or a pair of integers per dimension:

```{code-cell} ipython3
cfg = ps.CreateKernelConfig(
    ghost_layers=[(2, 1), 3]
)
```

```{code-cell} ipython3
:tags: [remove-input]

_show_ispace(cfg)
```

#### Iteration Slices

Using the `iteration_slice` option, we can assert much finer control on the kernel's iteration space
by specifying it using sequences of Python {py:class}`slice` objects.

We can quickly create those using `ps.make_slice`, using the `start:stop:step` slice notation.
The easiest case is to set the iteration space with fixed numerical limits:

```{code-cell} ipython3
cfg = ps.CreateKernelConfig(
    iteration_slice=ps.make_slice[3:-4, 9:14]
)
```

```{code-cell} ipython3
:tags: [remove-input]

_show_ispace(cfg)
```

##### Strided Iteration

It is also possible to set up a strided iteration that skips over a fixed number of elements.
The following example processes only every second line in $y$-direction, using the slice `::2`:

```{code-cell} ipython3
cfg = ps.CreateKernelConfig(
    iteration_slice=ps.make_slice[::2, 3:-3]
)
```

```{code-cell} ipython3
:tags: [remove-input]

_show_ispace(cfg)
```

(example_triangular_iteration)=
##### Triangular Iteration

Iteration slices are not limited to constant numerical values; they can be arbitrarily complex
*SymPy* expressions.
By using the counter symbol for the first dimension to define the iteration limits of the second,
we can produce a triangular iteration pattern:

```{code-cell} ipython3
y = ps.DEFAULTS.spatial_counters[0]
cfg = ps.CreateKernelConfig(
    iteration_slice=ps.make_slice[:, y:]
)
```

:::{warning}
This type of dependency is restricted by the ordering of the iteration space dimensions:
The limits of a dimension can only depend on the counters of dimensions that are *slower*
than itself.
The ordering of dimensions is determined by the memory layout of the kernels' fields;
see also the [section on memory layouts](#section_memory_layout).
:::

```{code-cell} ipython3
:tags: [remove-input]

_show_ispace(cfg)
```

##### Red-Black Iteration

Using a case distinction for the second dimension's start index, we can even produce
a checkerboard pattern, as required for e.g. red-black Gauss-Seidel-type smoothers.
We use the integer remainder ({any}`int_rem`) to distinguish between even- and odd-numbered rows,
set the start value accordingly using {any}`sp.Piecewise <sympy.functions.elementary.piecewise.Piecewise>`,
and use a step size of two:

$$
  start(y)=
    \begin{cases}
      0 & \quad \text{if } y \; \mathrm{rem} \; 2 = 0 \\ 
      1 & \quad \text{otherwise}
    \end{cases}
$$


```{code-cell} ipython3
from pystencils.sympyextensions.integer_functions import int_rem

y = ps.DEFAULTS.spatial_counters[0]
start = sp.Piecewise(
    (0, sp.Eq(int_rem(y, 2), 0)),
    (1, True)
)
cfg = ps.CreateKernelConfig(
    iteration_slice=ps.make_slice[:, start::2]
)
```

:::{warning}
The restrictions on dimension ordering of the triangular iteration example apply
to the checkerboard-iteration as well.
:::

```{code-cell} ipython3
:tags: [remove-input]

_show_ispace(cfg)
```

(section_memory_layout)=
## Memory Layout and Dimension Ordering

:::{admonition} Developer To Do
Briefly explain about field memory layouts, cache locality, coalesced memory accesses (on GPU and vector CPUs),
and the need for correct ordering of the dimensions (loop order on CPU, thread indexing order on GPU).
:::

(section_codegen_stages)=
## Advanced: Understanding the Stages of the Code Generator

While translating a set of symbolic definitions to a kernel, the code generator of pystencils
goes through a number of stages, gradually extending and transforming the AST.
Pystencils allows you to retrieve and inspect the intermediate results produced by the
code generator, in order to better understand the process of kernel translation.
This can be immensely helpful when tracking down bugs or trying to explain unexpected
output code.

To get access to the intermediate results, the code generator has to be invoked in a slightly different way.
Instead of just calling `create_kernel`, we directly create the so-called *driver* and instruct it to
store its intermediate ASTs:

```{code-cell} ipython3
:tags: [remove-cell]
u_src, u_dst, f = ps.fields("u_src, u_dst, f: float32[2D]")
h = sp.Symbol("h")

cfg = ps.CreateKernelConfig(
  target=ps.Target.X86_AVX512,
  default_dtype="float32",
)
cfg.cpu.openmp.enable = True
cfg.cpu.vectorize.enable = True
cfg.cpu.vectorize.assume_inner_stride_one = True

assignments = [
  ps.Assignment(
    u_dst[0,0], (h**2 * f[0, 0] + u_src[1, 0] + u_src[-1, 0] + u_src[0, 1] + u_src[0, -1]) / 4
  )
]
```

```{code-cell} ipython3
driver = ps.codegen.get_driver(cfg, retain_intermediates=True)
kernel = driver(assignments)
ps.inspect(driver.intermediates)
```
