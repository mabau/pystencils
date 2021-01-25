"""Module to generate stencil kernels in C or CUDA using sympy expressions and call them as Python functions"""
from . import sympy_gmpy_bug_workaround  # NOQA
from . import fd
from . import stencil as stencil
from .assignment import Assignment, assignment_from_stencil
from .data_types import TypedSymbol
from .datahandling import create_data_handling
from .display_utils import show_code, get_code_obj, get_code_str, to_dot
from .field import Field, FieldType, fields
from .kernel_decorator import kernel
from .kernelcreation import create_indexed_kernel, create_kernel, create_staggered_kernel
from .simp import AssignmentCollection
from .slicing import make_slice
from .sympyextensions import SymbolCreator
from .spatial_coordinates import (x_, x_staggered, x_staggered_vector, x_vector,
                                  y_, y_staggered, z_, z_staggered)

try:
    import pystencils_autodiff
    autodiff = pystencils_autodiff
except ImportError:
    pass


def _get_release_file():
    import os.path
    file_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(file_path, '..', 'RELEASE-VERSION')


try:
    __version__ = open(_get_release_file(), 'r').read()
except IOError:
    __version__ = 'development'

__all__ = ['Field', 'FieldType', 'fields',
           'TypedSymbol',
           'make_slice',
           'create_kernel', 'create_indexed_kernel', 'create_staggered_kernel',
           'show_code', 'to_dot', 'get_code_obj', 'get_code_str',
           'AssignmentCollection',
           'Assignment',
           'assignment_from_stencil',
           'SymbolCreator',
           'create_data_handling',
           'kernel',
           'x_', 'y_', 'z_',
           'x_staggered', 'y_staggered', 'z_staggered',
           'x_vector', 'x_staggered_vector',
           'fd',
           'stencil']
