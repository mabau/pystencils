import os
import sys
import io
from setuptools import setup, find_packages
import distutils
from contextlib import redirect_stdout
from importlib import import_module
sys.path.insert(0, os.path.abspath('doc'))
from version_from_git import version_number_from_git


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


setup(name='pystencils',
      version=version_number_from_git(),
      description='Speeding up stencil computations on CPUs and GPUs',
      long_description=readme(),
      long_description_content_type="text/markdown",
      author='Martin Bauer',
      license='AGPLv3',
      author_email='martin.bauer@fau.de',
      url='https://i10git.cs.fau.de/software/pystencils/',
      packages=['pystencils'] + ['pystencils.' + s for s in find_packages('pystencils')],
      install_requires=['sympy>=1.1', 'numpy', 'appdirs', 'joblib'],
      package_data={'pystencils': ['include/*.h']},
      classifiers=[
          'Development Status :: 4 - Beta',
          'Framework :: Jupyter',
          'Topic :: Software Development :: Code Generators',
          'Topic :: Scientific/Engineering :: Physics',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
      ],
      extras_require={
          'gpu': ['pycuda'],
          'alltrafos': ['islpy', 'py-cpuinfo'],
          'bench_db': ['blitzdb', 'pymongo', 'pandas'],
          'interactive': ['matplotlib', 'ipy_table', 'imageio', 'jupyter', 'pyevtk'],
          'doc': ['sphinx', 'sphinx_rtd_theme', 'nbsphinx',
                  'sphinxcontrib-bibtex', 'sphinx_autodoc_typehints', 'pandoc'],
      },
      tests_require=['pytest', 'pytest-cov', 'pytest-xdist', 'flake8', 'nbformat', 'nbconvert', 'ipython'],
      python_requires=">=3.6",
      cmdclass={
          'quicktest': SimpleTestRunner
      }
      )
