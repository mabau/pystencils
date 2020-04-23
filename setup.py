import distutils
import io
import os
import sys
from contextlib import redirect_stdout
from importlib import import_module

from setuptools import find_packages, setup

if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False

quick_tests = [
    'test_datahandling.test_kernel',
    'test_blocking_staggered.test_blocking_staggered',
    'test_blocking_staggered.test_blocking_staggered',
    'test_vectorization.test_vectorization_variable_size',
]


class SimpleTestRunner(distutils.cmd.Command):
    """A custom command to run selected tests"""

    description = 'run some quick tests'
    user_options = []

    @staticmethod
    def _run_tests_in_module(test):
        """Short test runner function - to work also if py.test is not installed."""
        test = 'pystencils_tests.' + test
        mod, function_name = test.rsplit('.', 1)
        if isinstance(mod, str):
            mod = import_module(mod)

        func = getattr(mod, function_name)
        print("   -> %s in %s" % (function_name, mod.__name__))
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
    ext = '.pyx' if USE_CYTHON else '.c'
    result = [Extension(e, [e.replace('.', '/') + ext]) for e in extensions]
    if USE_CYTHON:
        from Cython.Build import cythonize
        result = cythonize(result, language_level=3)
    return result


try:
    sys.path.insert(0, os.path.abspath('doc'))
    from version_from_git import version_number_from_git

    version = version_number_from_git()
    with open("RELEASE-VERSION", "w") as f:
        f.write(version)
except ImportError:
    version = open('RELEASE-VERSION', 'r').read()

setup(name='pystencils',
      description='Speeding up stencil computations on CPUs and GPUs',
      version=version,
      long_description=readme(),
      long_description_content_type="text/markdown",
      author='Martin Bauer',
      license='AGPLv3',
      author_email='martin.bauer@fau.de',
      url='https://i10git.cs.fau.de/pycodegen/pystencils/',
      packages=['pystencils'] + ['pystencils.' + s for s in find_packages('pystencils')],
      install_requires=['sympy>=1.1', 'numpy', 'appdirs', 'joblib'],
      package_data={'pystencils': ['include/*.h',
                                   'backends/cuda_known_functions.txt',
                                   'backends/opencl1.1_known_functions.txt']},

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
          'opencl': ['pyopencl'],
          'alltrafos': ['islpy', 'py-cpuinfo'],
          'bench_db': ['blitzdb', 'pymongo', 'pandas'],
          'interactive': ['matplotlib', 'ipy_table', 'imageio', 'jupyter', 'pyevtk', 'rich', 'graphviz'],
          'autodiff': ['pystencils-autodiff'],
          'doc': ['sphinx', 'sphinx_rtd_theme', 'nbsphinx',
                  'sphinxcontrib-bibtex', 'sphinx_autodoc_typehints', 'pandoc'],
      },
      tests_require=['pytest',
                     'pytest-cov',
                     'pytest-html',
                     'ansi2html',
                     'pytest-xdist',
                     'flake8',
                     'nbformat',
                     'nbconvert',
                     'ipython'],

      python_requires=">=3.6",
      cmdclass={
          'quicktest': SimpleTestRunner
      },
      )
