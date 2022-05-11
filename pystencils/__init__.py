"""Module to generate stencil kernels in C or CUDA using sympy expressions and call them as Python functions"""
from .enums import Backend, Target
from . import fd
from . import stencil as stencil
from .assignment import Assignment, assignment_from_stencil
from pystencils.typing.typed_sympy import TypedSymbol
from .datahandling import create_data_handling
from .display_utils import get_code_obj, get_code_str, show_code, to_dot
from .field import Field, FieldType, fields
from .config import CreateKernelConfig
from .kernel_decorator import kernel, kernel_config
from .kernelcreation import create_kernel, create_staggered_kernel
from .simp import AssignmentCollection
from .slicing import make_slice
from .spatial_coordinates import x_, x_staggered, x_staggered_vector, x_vector, y_, y_staggered, z_, z_staggered
from .sympyextensions import SymbolCreator

__all__ = ['Field', 'FieldType', 'fields',
           'TypedSymbol',
           'make_slice',
           'CreateKernelConfig',
           'create_kernel', 'create_staggered_kernel',
           'Target', 'Backend',
           'show_code', 'to_dot', 'get_code_obj', 'get_code_str',
           'AssignmentCollection',
           'Assignment',
           'assignment_from_stencil',
           'SymbolCreator',
           'create_data_handling',
           'kernel', 'kernel_config',
           'x_', 'y_', 'z_',
           'x_staggered', 'y_staggered', 'z_staggered',
           'x_vector', 'x_staggered_vector',
           'fd',
           'stencil']

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
