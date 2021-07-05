# Contributing

Contributions to pystencils are always welcome, and they are greatly appreciated!
A list of open problems can be found [here]( https://i10git.cs.fau.de/pycodegen/pystencils/-/issues).
Of course, it is also always appreciated to bring own ideas and problems to the community!


Please submit all contributions to the official [GitLab repository](https://i10git.cs.fau.de/pycodegen/pystencils) in the form of a Merge Request. Please do not submit git diffs or files containing the changes.
There also exists a GitHub repository, which is only a mirror to the GitLab repository. Contributions to the GitHub repository are not considered.

`pystencils` is an open-source python package under the license of  AGPL3. Thus we consider the act of contributing to the code by submitting a Merge Request as the "Sign off" or agreement to the AGPL3 license.

You can contribute in many different ways:

## Types of Contributions

### Report Bugs

Report bugs at [https://i10git.cs.fau.de/pycodegen/pystencils/-/issues](https://i10git.cs.fau.de/pycodegen/pystencils/-/issues).

For pystencils, it is often necessary to provide the python and [SymPy](https://www.sympy.org/en/index.html) versions used and hardware information like the
processor architecture and the compiler version used to compile the generated kernels.

### Fix Issues

Look through the GitLab issues. Different tags are indicating the status of the issues.
The "bug" tag indicates problems with pystencils, while the "feature" tag shows ideas that should be added in the future.

### Write Documentation

The documentation of pystencils can be found [here](https://pycodegen.pages.i10git.cs.fau.de/pystencils/). Jupyter notebooks are used to provide an
interactive start to pystencils. It is always appreciated if new document notebooks are provided
since this helps others a lot.

## Get Started!

Ready to contribute? Here is how to set up `pystencils` for local development.

1. Fork the `pystencils` repo on GitLab.
2. Clone your fork locally:
```bash
    $ git clone https://i10git.cs.fau.de/your-name/pystencils
```
3. Install your local copy into a virtualenv. It is also recommended to use anaconda or miniconda to manage the python environments.
```bash
    $ mkvirtualenv pystencils
    $ cd pystencils/
    $ pip install -e .
```
4. Create a branch for local development:
```bash
    $ git checkout -b name-of-your-bugfix-or-feature
```
   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests
```bash
    $ flake8 pystencils
    $ py.test -v -n $NUM_CORES -m "not longrun" .
   
```

   To get all packages needed for development, a requirements list can be found [here](https://i10git.cs.fau.de/pycodegen/pycodegen/-/blob/master/conda_environment_dev.yml). This includes flake8 and pytest.

6. Commit your changes and push your branch to GitHub::
```bash
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
```
7. Submit a Merge Request on GitLab.

## Merge Request Guidelines

Before you submit a Merge Request, check that it meets these guidelines:

1. All functionality that is implemented through this Merge Request should be covered by unit tests. These are implemented in `pystencil_tests`
2. If the Merge Request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring.
3. If you have a maintainer status for `pystencils`, you can merge Merge Requests to the master branch. However, every Merge Request needs to be reviewed by another developer. Thus it is not allowed to merge a Merge Request, which is submitted by oneself.

## Tips

To run a subset of tests:
```bash
$ py.test my_test.py
```