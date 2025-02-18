(_installation)=
# Setup and Installation

## Install pystencils

There are two ways to install the latest development version of pystencils 2.0.
You can either install it directly from our git repository:

```bash
pip install "git+https://i10git.cs.fau.de/pycodegen/pystencils.git@v2.0-dev"
```

Or clone the repository locally and perform an editable install:

```bash
git clone -b v2.0-dev https://i10git.cs.fau.de/pycodegen/pystencils.git
pip install -e pystencils
```

### Feature Groups

In both cases, you can add a set of optional features to your installation by listing them
in square brackets (e.g. `pip install -e pystencils[feature1, feature2]`).
The following feature sets are available:

- `interactive` (**recommended**): Install dependencies for using pystencils interactively from
  within Jupyter notebooks.
  Setting this flag will cause pip to install `jupyter`, `matplotlib`, and `graphviz`, among others, alongside pystencils.
- `alltrafos` (**recommended**): Install dependencies to enable a wider set of code transformation.
  These include [islpy](https://pypi.org/project/islpy/) for polyhedral loop transformations,
  and [py-cpuinfo](https://pypi.org/project/py-cpuinfo/) for detecting the current hardware in order
  to select optimal vector instructions.
- `use_cython`: Install [Cython](https://cython.org/), which is used internally by pystencils
  to accelerate the setup of boundary conditions.

:::{dropdown} For Developers

If you are developing pystencils, we recommend you perform an editable install of your
local clone of the repository, with all optional features:
```bash
pip install -e pystencils[alltrafos,interactive,use_cython,doc,testsuite]
```

This includes the additional feature groups `doc`, which contains all dependencies required
to build this documentation, and `tests`, which adds `flake8` for code style checking,
`mypy` for static type checking, and `pytest` plus plugins for running the test suite.

For more information on developing pystencils, see the [](#contribution_guide).
:::

### For Nvidia GPUs

If you have an Nvidia graphics processor and CUDA installed, you can use pystencils to directly compile
and execute kernels running on your GPU.
This requires a working installation of [cupy](https://cupy.dev).
Please refer to the cupy's [installation manual](https://docs.cupy.dev/en/stable/install.html)
for details about installing cupy.
