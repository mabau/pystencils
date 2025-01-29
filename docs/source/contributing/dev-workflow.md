# Development Workflow

This page contains instructions on how to get started with developing pystencils.

## Prepare the Git Repository

The pystencils Git repository is hosted at [i10git.cs.fau.de](https://i10git.cs.fau.de), the GitLab instance of the
[Chair for Systems Simulation](https://www.cs10.tf.fau.de/) at [FAU Erlangen-Nürnberg](https://fau.de).
In order to contribute code to pystencils, you will need to acquire an account there; to do so,
please follow the instructions on the GitLab landing page.

### Create a Fork

Only the core developers of pystencils have write-access to the primary repository.
To contribute, you will therefore have to create a fork of that repository
by navigating to the [repository page](https://i10git.cs.fau.de/pycodegen/pystencils)
and selecting *Fork* there.
In this fork, you may freely create branches and develop code, which may later be merged to a primary branch
via merge requests.

### Create a Local Clone

Once you have a fork of the repository, you can clone it to your local machine using the git command-line.

:::{note}
To clone via SSH, which is recommended, you will first have to [register an SSH key](https://docs.gitlab.com/ee/user/ssh.html).
:::

Open up a shell and navigate to a directory you want to work in.
Then, enter

```bash
git clone git@i10git.cs.fau.de:<your-username>/pystencils.git
```

to clone your fork of pystencils.

:::{note}
To keep up to date with the upstream repository, you can add it as a secondary remote to your clone:
```bash
git remote add upstream git@i10git.cs.fau.de:pycodegen/pystencils.git
```
You can point your clone's `master` branch at the upstream master like this:
```bash
git pull --set-upstream upstream master
```
:::

## Set Up the Python Environment

To develop pystencils, you will need at least the following software installed on your machine:

- Python 3.10 or later: Since pystencils minimal supported version is Python 3.10, we recommend that you work with Python 3.10 directly.
- An up-to-date C++ compiler, used by pystencils to JIT-compile generated code
- [Nox](https://nox.thea.codes/en/stable/), which we use for test automation.
  Nox will be used extensively in the instructions on testing below.
- Optionally [CUDA](https://developer.nvidia.com/cuda-toolkit),
  if you have an Nvidia or AMD GPU and plan to develop on pystencils' GPU capabilities

Once you have these, set up a [virtual environment](https://docs.python.org/3/library/venv.html) for development.
This ensures that your system's installation of Python is kept clean, and isolates your development environment
from outside influence.
Use the following commands to create a virtual environment at `.venv` and perform an editable install of pystencils into it:

```bash
python -m venv .venv
source .venv/bin/activate
export PIP_REQUIRE_VIRTUALENV=true
pip install -e .[dev]
```

:::{note}
Setting `PIP_REQUIRE_VIRTUALENV` ensures that pip refuses to install packages globally --
Consider setting this variable globally in your shell's configuration file.
:::

You are now ready to go! Create a new git branch to work on, open up an IDE, and start coding.
Make sure your IDE recognizes the virtual environment you created, though.

## Static Code Analysis

### PEP8 Code Style

We use [flake8](https://github.com/PyCQA/flake8/tree/main) to check our code for compliance with the
[PEP8](https://peps.python.org/pep-0008/) code style.
You can either run `flake8` directly, or through Nox, to analyze your code with respect to style errors:

::::{grid}
:::{grid-item}
```bash
nox -s lint
```
:::
:::{grid-item}
```bash
flake8 src/pystencils
```
:::
::::

### Static Type Checking

New code added to pystencils is required to carry type annotations,
and its types are checked using [mypy](https://mypy.readthedocs.io/en/stable/index.html#).
To discover type errors, run *mypy* either directly or via Nox:

::::{grid}
:::{grid-item}
```bash
nox -s typecheck
```
:::
:::{grid-item}
```bash
mypy src/pystencils
```
:::
::::

:::{note}
Type checking is currently restricted only to a few modules, which are listed in the `mypy.ini` file.
Most code in the remaining modules is significantly older and is not comprehensively type-annotated.
As more modules are updated with type annotations, this list will expand in the future.
If you think a new module is ready to be type-checked, add an exception clause to `mypy.ini`.
:::

## Running the Test Suite

Pystencils comes with an extensive and steadily growing suite of unit tests.
To run the testsuite, you may invoke a variant of the Nox `testsuite` session.
There are multiple different versions of the `testsuite` session, depending on whether you are testing with our
without CUDA, or which version of Python you wish to test with.
You can list the available sessions using `nox -l`.
Select one of the `testsuite` variants and run it via `nox -s "testsuite(<variant>)"`, e.g.
```
nox -s "testsuite(cpu)"
```
for the CPU-only suite.

During the testsuite run, coverage information is collected and displayed using [coverage.py](https://coverage.readthedocs.io/en/7.6.10/).
You can display a detailed overview of code coverage by opening the generated `htmlcov/index.html` page.

## Building the Documentation

The pystencils documentation pages are written in MyST Markdown and ReStructuredText,
located at the `docs` folder, and built using Sphinx.
To build the documentation pages of pystencils, simply run the `docs` Nox session:
```bash
nox -s docs
```

This will emit the generated HTML pages to `docs/build/html`.
The `docs` session permits two parameters to customize its execution:
 - `--clean`: Clean the page generator's output before building
 - `--fail-on-warnings`: Have the build fail (finish with a nonzero exit code) if Sphinx emits any warnings.

You must pass any of these to the session command *after a pair of dashes* (`--`); e.g.:
```bash
nox -s docs -- --clean
```
