import os
import sys
from setuptools import setup, find_packages
sys.path.insert(0, os.path.abspath('..'))
from custom_pypi_index.pypi_index import get_current_dev_version_from_git


setup(name='pystencils',
      version=get_current_dev_version_from_git(),
      description='Python Stencil Compiler based on sympy as numpy',
      author='Martin Bauer',
      license='AGPLv3',
      author_email='martin.bauer@fau.de',
      url='https://i10git.cs.fau.de/software/pystencils/',
      packages=['pystencils'] + ['pystencils.' + s for s in find_packages('pystencils')],
      install_requires=['sympy>=1.1', 'numpy', 'appdirs', 'joblib'],
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
      tests_require=['pytest', 'pytest-cov', 'pytest-xdist', 'flake8'],
      python_requires=">=3.6",
      )
