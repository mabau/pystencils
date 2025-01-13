"""Module to generate stencil kernels in C or CUDA using sympy expressions and call them as Python functions"""

from .codegen import (
    Target,
    CreateKernelConfig,
    CpuOptimConfig,
    VectorizationConfig,
    OpenMpConfig,
    GpuIndexingConfig,
    AUTO
)
from .defaults import DEFAULTS
from . import fd
from . import stencil as stencil
from .display_utils import get_code_obj, get_code_str, show_code, to_dot
from .inspection import inspect
from .field import Field, FieldType, fields
from .types import create_type, create_numeric_type
from .cache import clear_cache
from .kernel_decorator import kernel, kernel_config
from .kernelcreation import create_kernel, create_staggered_kernel
from .codegen import Kernel
from .jit import no_jit
from .backend.exceptions import KernelConstraintsError
from .slicing import make_slice
from .spatial_coordinates import (
    x_,
    x_staggered,
    x_staggered_vector,
    x_vector,
    y_,
    y_staggered,
    z_,
    z_staggered,
)
from .assignment import Assignment, AddAugmentedAssignment, assignment_from_stencil
from .simp import AssignmentCollection
from .sympyextensions.typed_sympy import TypedSymbol, DynamicType
from .sympyextensions import SymbolCreator
from .datahandling import create_data_handling

__all__ = [
    "Field",
    "FieldType",
    "fields",
    "DEFAULTS",
    "TypedSymbol",
    "DynamicType",
    "create_type",
    "create_numeric_type",
    "make_slice",
    "CreateKernelConfig",
    "CpuOptimConfig",
    "VectorizationConfig",
    "GpuIndexingConfig",
    "OpenMpConfig",
    "AUTO",
    "create_kernel",
    "create_staggered_kernel",
    "Kernel",
    "KernelConstraintsError",
    "Target",
    "no_jit",
    "show_code",
    "to_dot",
    "get_code_obj",
    "get_code_str",
    "inspect",
    "AssignmentCollection",
    "Assignment",
    "AddAugmentedAssignment",
    "assignment_from_stencil",
    "SymbolCreator",
    "create_data_handling",
    "clear_cache",
    "kernel",
    "kernel_config",
    "x_",
    "y_",
    "z_",
    "x_staggered",
    "y_staggered",
    "z_staggered",
    "x_vector",
    "x_staggered_vector",
    "fd",
    "stencil",
]

from . import _version
__version__ = _version.get_versions()['version']
