import distutils
import io
import os
from contextlib import redirect_stdout
from importlib import import_module

import setuptools

import versioneer

try:
    import cython  # noqa

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

quick_tests = [
    'test_quicktests.test_basic_kernel',
    'test_quicktests.test_basic_blocking_staggered',
    'test_quicktests.test_basic_vectorization',
]


class SimpleTestRunner(distutils.cmd.Command):
    """A custom command to run selected tests"""

    description = 'run some quick tests'
    user_options = []

    @staticmethod
    def _run_tests_in_module(test):
        """Short test runner function - to work also if py.test is not installed."""
        test = f'pystencils_tests.{test}'
        mod, function_name = test.rsplit('.', 1)
        if isinstance(mod, str):
            mod = import_module(mod)

        func = getattr(mod, function_name)
        print(f"   -> {function_name} in {mod.__name__}")
        with redirect_stdout(io.StringIO()):
            func()

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run command."""
        for test in quick_tests:
            self._run_tests_in_module(test)


def readme():
    with open('README.md') as f:
        return f.read()


def cython_extensions(*extensions):
    from distutils.extension import Extension
    if USE_CYTHON:
        ext = '.pyx'
        result = [Extension(e, [os.path.join(*e.split(".")) + ext]) for e in extensions]
        from Cython.Build import cythonize
        result = cythonize(result, language_level=3)
        return result
    elif all([os.path.exists(os.path.join(*e.split(".")) + '.c') for e in extensions]):
        ext = '.c'
        result = [Extension(e, [os.path.join(*e.split(".")) + ext]) for e in extensions]
        return result
    else:
        return None


def get_cmdclass():
    cmdclass = {"quicktest": SimpleTestRunner}
    cmdclass.update(versioneer.get_cmdclass())
    return cmdclass


setuptools.setup(name='pystencils',
                 description='Speeding up stencil computations on CPUs and GPUs',
                 version=versioneer.get_version(),
                 long_description=readme(),
                 long_description_content_type="text/markdown",
                 author='Martin Bauer, Jan Hönig, Markus Holzer',
                 license='AGPLv3',
                 author_email='cs10-codegen@fau.de',
                 url='https://i10git.cs.fau.de/pycodegen/pystencils/',
                 packages=['pystencils'] + ['pystencils.' + s for s in setuptools.find_packages('pystencils')],
                 install_requires=['sympy>=1.6,<=1.11.1', 'numpy>=1.8.0', 'appdirs', 'joblib'],
                 package_data={'pystencils': ['include/*.h',
                                              'backends/cuda_known_functions.txt',
                                              'backends/opencl1.1_known_functions.txt',
                                              'boundaries/createindexlistcython.c',
                                              'boundaries/createindexlistcython.pyx']},
                 ext_modules=cython_extensions("pystencils.boundaries.createindexlistcython"),
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'Framework :: Jupyter',
                     'Topic :: Software Development :: Code Generators',
                     'Topic :: Scientific/Engineering :: Physics',
                     'Intended Audience :: Developers',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
                 ],
                 project_urls={
                     "Bug Tracker": "https://i10git.cs.fau.de/pycodegen/pystencils/issues",
                     "Documentation": "http://pycodegen.pages.walberla.net/pystencils/",
                     "Source Code": "https://i10git.cs.fau.de/pycodegen/pystencils",
                 },
                 extras_require={
                     'gpu': ['pycuda'],
                     'alltrafos': ['islpy', 'py-cpuinfo'],
                     'bench_db': ['blitzdb', 'pymongo', 'pandas'],
                     'interactive': ['matplotlib', 'ipy_table', 'imageio', 'jupyter', 'pyevtk', 'rich', 'graphviz'],
                     'doc': ['sphinx', 'sphinx_rtd_theme', 'nbsphinx',
                             'sphinxcontrib-bibtex', 'sphinx_autodoc_typehints', 'pandoc'],
                     'use_cython': ['Cython']
                 },
                 tests_require=['pytest',
                                'pytest-cov',
                                'pytest-html',
                                'ansi2html',
                                'pytest-xdist',
                                'flake8',
                                'nbformat',
                                'nbconvert',
                                'ipython',
                                'randomgen>=1.18'],

                 python_requires=">=3.8",
                 cmdclass=get_cmdclass()
                 )
