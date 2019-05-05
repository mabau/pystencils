"""Module to generate stencil kernels in C or CUDA using sympy expressions and call them as Python functions"""
from . import sympy_gmpy_bug_workaround  # NOQA
from .field import Field, FieldType, fields
from .data_types import TypedSymbol
from .slicing import make_slice
from .kernelcreation import create_kernel, create_indexed_kernel, create_staggered_kernel
from .display_utils import show_code, to_dot
from .simp import AssignmentCollection
from .assignment import Assignment, assignment_from_stencil
from .sympyextensions import SymbolCreator
from .datahandling import create_data_handling
from .kernel_decorator import kernel
from . import fd
from . import stencil as stencil


__all__ = ['Field', 'FieldType', 'fields',
           'TypedSymbol',
           'make_slice',
           'create_kernel', 'create_indexed_kernel', 'create_staggered_kernel',
           'show_code', 'to_dot',
           'AssignmentCollection',
           'Assignment',
           'assignment_from_stencil',
           'SymbolCreator',
           'create_data_handling',
           'kernel',
           'fd',
           'stencil']
