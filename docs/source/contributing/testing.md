(testing_pystencils)=
# Testing pystencils

The pystencils testsuite is located at the `tests` directory,
constructed using [pytest](https://pytest.org),
and automated through [Nox](https://nox.thea.codes).
On this page, you will find instructions on how to execute and extend it.

## Running the Testsuite

The fastest way to execute the pystencils test suite is through the `testsuite` Nox session:

```bash
nox -s testsuite
```

There exist several configurations of the testsuite session, from which the above command will
select and execute only those that are available on your machine.
 - *Python Versions:* The testsuite session can be run against all major Python versions between 3.10 and 3.13 (inclusive).
   To only use a specific Python version, add the `-p 3.XX` argument to your Nox invocation; e.g. `nox -s testsuite -p 3.11`.
 - *CuPy:* There exist three variants of `testsuite`, including or excluding tests for the CUDA GPU target: `cpu`, `cupy12` and `cupy13`.
   To select one, append `(<variant>)` to the session name; e.g. `nox -s "testsuite(cupy12)"`.

You may also pass options through to pytest via positional arguments after a pair of dashes, e.g.:

```bash
nox -s testsuite -- -k "kernelcreation"
```

During the testsuite run, coverage information is collected using [coverage.py](https://coverage.readthedocs.io/en/7.6.10/),
and the results are exported to HTML.
You can display a detailed overview of code coverage by opening the generated `htmlcov/index.html` page.

## Extending the Test Suite

### Codegen Configurations via Fixtures

In the pystencils test suite, it is often necessary to test code generation features against multiple targets.
To simplify this process, we provide a number of [pytest fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
you can and should use in your tests:

 - `target`: Provides code generation targets for your test.
   Using this fixture will make pytest create a copy of your test for each target
   available on the current machine (see {any}`Target.available_targets`).
 - `gen_config`: Provides default code generation configurations for your test.
   This fixture depends on `target` and provides a {any}`CreateKernelConfig` instance
   with target-specific optimization options (in particular vectorization) enabled.
 - `xp`: The `xp` fixture gives you either the *NumPy* (`np`) or the *CuPy* (`cp`) module,
   depending on whether `target` is a CPU or GPU target.

These fixtures are defined in `tests/fixtures.py`.

### Overriding Fixtures

Pytest allows you to locally override fixtures, which can be especially handy when you wish
to restrict the target selection of a test.
For example, the following test overrides `target` using a parametrization mark,
and uses this in combination with the `gen_config` fixture, which now
receives the overridden `target` parameter as input:

```Python
@pytest.mark.parametrize("target", [Target.X86_SSE, Target.X86_AVX])
def test_bogus(gen_config):
    assert gen_config.target.is_vector_cpu()
```

## Testing with the Experimental CPU JIT

Currently, the testsuite by default still uses the {any}`legacy CPU JIT compiler <LegacyCpuJit>`,
since the new CPU JIT compiler is still in an experimental stage.
To test your code against the new JIT compiler, pass the `--experimental-cpu-jit` option to pytest:

```bash
nox -s testsuite -- --experimental-cpu-jit
```

This will alter the `gen_config` fixture, activating the experimental CPU JIT for CPU targets.
