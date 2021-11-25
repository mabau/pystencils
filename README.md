pystencils
==========

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mabau/pystencils/master?filepath=doc%2Fnotebooks)
[![Docs](https://img.shields.io/badge/read-the_docs-brightgreen.svg)](https://pycodegen.pages.i10git.cs.fau.de/pystencils)
[![pypi-package](https://badge.fury.io/py/pystencils.svg)](https://badge.fury.io/py/pystencils)
[![pipeline status](https://i10git.cs.fau.de/pycodegen/pystencils/badges/master/pipeline.svg)](https://i10git.cs.fau.de/pycodegen/pystencils/commits/master)
[![coverage report](https://i10git.cs.fau.de/pycodegen/pystencils/badges/master/coverage.svg)](http://pycodegen.pages.i10git.cs.fau.de/pystencils/coverage_report)

Run blazingly fast stencil codes on numpy arrays.

*pystencils* uses sympy to define stencil operations, that can be executed on numpy arrays.
Exploiting the stencil structure makes *pystencils* run faster than normal numpy code and even as Cython and numba,
[as demonstrated in this notebook](https://pycodegen.pages.i10git.cs.fau.de/pystencils/notebooks/demo_benchmark.html).


Here is a code snippet that computes the average of neighboring cells:
```python
import pystencils as ps
import numpy as np

f, g = ps.fields("f, g : [2D]")
stencil = ps.Assignment(g[0, 0],
                       (f[1, 0] + f[-1, 0] + f[0, 1] + f[0, -1]) / 4)
kernel = ps.create_kernel(stencil).compile()

f_arr = np.random.rand(1000, 1000)
g_arr = np.empty_like(f_arr)
kernel(f=f_arr, g=g_arr)
```

*pystencils* is mostly used for numerical simulations using finite difference or finite volume methods.
It comes with automatic finite difference discretization for PDEs:

```python
import pystencils as ps
import sympy as sp

c, v = ps.fields("c, v(2): [2D]")
adv_diff_pde = ps.fd.transient(c) - ps.fd.diffusion(c, sp.symbols("D")) + ps.fd.advection(c, v)
discretize = ps.fd.Discretization2ndOrder(dx=1, dt=0.01)
discretization = discretize(adv_diff_pde)
```

Installation
------------

```bash
pip install pystencils[interactive]
```

Without `[interactive]` you get a minimal version with very little dependencies.

All options:
- `gpu`: use this if an NVIDIA GPU is available and CUDA is installed
- `alltrafos`: pulls in additional dependencies for loop simplification e.g. libisl
- `bench_db`: functionality to store benchmark result in object databases
- `interactive`: installs dependencies to work in Jupyter including image I/O, plotting etc.
- `doc`: packages to build documentation

Options can be combined e.g.
```bash
pip install pystencils[interactive, gpu, doc]
```

pystencils is also fully compatible with Windows machines. If working with visual studio and pycuda makes sure to run example files first to ensure that pycuda can find the compiler's executable.

Documentation
-------------

Read the docs [here](https://pycodegen.pages.i10git.cs.fau.de/pystencils) and
check out the Jupyter notebooks in `doc/notebooks`. The **Changelog** of pystencils can be found [here](https://i10git.cs.fau.de/pycodegen/pystencils/-/blob/master/CHANGELOG.md).

Authors
-------

Many thanks go to the [contributors](https://i10git.cs.fau.de/pycodegen/pystencils/-/blob/master/AUTHORS.txt) of pystencils.

### Please cite us

If you use pystencils in a publication, please cite the following articles:

Overview:
  - M. Bauer et al, Code Generation for Massively Parallel Phase-Field Simulations. Association for Computing Machinery, 2019. https://doi.org/10.1145/3295500.3356186