import warnings
from copy import copy
from collections import defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Union, Tuple, List, Dict, Callable, Any

from pystencils import Target, Backend, Field
from pystencils.typing.typed_sympy import BasicType

import numpy as np


# TODO: CreateKernelConfig is bloated think of more classes better usage, factory whatever ...
# Proposition: CreateKernelConfigs Classes for different targets?
@dataclass
class CreateKernelConfig:
    """
    **Below all parameters for the CreateKernelConfig are explained**
    """
    target: Target = Target.CPU
    """
    All targets are defined in :class:`pystencils.enums.Target`
    """
    backend: Backend = None
    """
    All backends are defined in :class:`pystencils.enums.Backend`
    """
    function_name: str = 'kernel'
    """
    Name of the generated function - only important if generated code is written out
    """
    # TODO Sane defaults: config should check that the datatype is a Numpy type
    # TODO Sane defaults: QoL default_number_float and default_number_int should be data_type if they are not specified
    data_type: Union[str, Dict[str, BasicType]] = 'float64'
    """
    Data type used for all untyped symbols (i.e. non-fields), can also be a dict from symbol name to type
    """
    default_number_float: Union[str, np.dtype, BasicType] = 'float64'
    """
    Data type used for all untyped floating point numbers (i.e. 0.5)
    """
    default_number_int: Union[str, np.dtype, BasicType] = 'int64'
    """
    Data type used for all untyped integer numbers (i.e. 1)
    """
    iteration_slice: Tuple = None
    """
    Rectangular subset to iterate over, if not specified the complete non-ghost layer part of the field is iterated over
    """
    ghost_layers: Union[bool, int, List[Tuple[int]]] = None
    """
    A single integer specifies the ghost layer count at all borders, can also be a sequence of
    pairs ``[(x_lower_gl, x_upper_gl), .... ]``. These layers are excluded from the iteration.
    If left to default, the number of ghost layers is determined automatically from the assignments.
    """
    cpu_openmp: Union[bool, int] = False
    """
    `True` or number of threads for OpenMP parallelization, `False` for no OpenMP. If set to `True`, the maximum number
    of available threads will be chosen.
    """
    cpu_vectorize_info: Dict = None
    """
    A dictionary with keys, 'vector_instruction_set', 'assume_aligned' and 'nontemporal'
    for documentation of these parameters see vectorize function. Example:
    '{'instruction_set': 'avx512', 'assume_aligned': True, 'nontemporal':True}'
    """
    cpu_blocking: Tuple[int] = None
    """
    A tuple of block sizes or `None` if no blocking should be applied
    """
    omp_single_loop: bool = True
    """
    If OpenMP is active: whether multiple outer loops are permitted
    """
    gpu_indexing: str = 'block'
    """
    Either 'block' or 'line' , or custom indexing class, see `pystencils.gpucuda.AbstractIndexing`
    """
    gpu_indexing_params: MappingProxyType = field(default=MappingProxyType({}))
    """
    Dict with indexing parameters (constructor parameters of indexing class)
    e.g. for 'block' one can specify '{'block_size': (20, 20, 10) }'.
    """
    # TODO Markus rework this docstring
    default_assignment_simplifications: bool = False
    """
    If `True` default simplifications are first performed on the Assignments. If problems occur during the
    simplification a warning will be thrown.
    Furthermore, it is essential to know that this is a two-stage process. The first stage of the process acts
    on the level of the `pystencils.AssignmentCollection`.  In this part,
    `pystencil.simp.create_simplification_strategy` from pystencils.simplificationfactory will be used to
    apply optimisations like insertion of constants to
    remove pressure from the registers. Thus the first part of the optimisations can only be executed if
    an `AssignmentCollection` is passed. The second part of the optimisation acts on the level of each Assignment
    individually. In this stage, all optimisations from `sympy.codegen.rewriting.optims_c99` are applied
    to each Assignment. Thus this stage can also be applied if a list of Assignments is passed.
    """
    cpu_prepend_optimizations: List[Callable] = field(default_factory=list)
    """
    List of extra optimizations to perform first on the AST.
    """
    use_auto_for_assignments: bool = False
    """
    If set to `True`, auto can be used in the generated code for data types. This makes the type system more robust.
    """
    index_fields: List[Field] = None
    """
    List of index fields, i.e. 1D fields with struct data type. If not `None`, `create_index_kernel`
    instead of `create_domain_kernel` is used.
    """
    coordinate_names: Tuple[str, Any] = ('x', 'y', 'z')
    """
    Name of the coordinate fields in the struct data type.
    """
    allow_double_writes: bool = False
    """
    If True, don't check if every field is only written at a single location. This is required
    for example for kernels that are compiled with loop step sizes > 1, that handle multiple 
    cells at once. Use with care!
    """
    skip_independence_check: bool = False
    """
    Don't check that loop iterations are independent. This is needed e.g. for 
    periodicity kernel, that access the field outside the iteration bounds. Use with care!
    """

    class DataTypeFactory:
        """Because of pickle, we need to have a nested class, instead of a lambda in __post_init__"""
        def __init__(self, dt):
            self.dt = dt

        def __call__(self):
            return BasicType(self.dt)

    def __post_init__(self):
        # ----  Legacy parameters
        # TODO Sane defaults: Check for abmigous types like "float", python float, which are dangerous for users
        if isinstance(self.target, str):
            new_target = Target[self.target.upper()]
            warnings.warn(f'Target "{self.target}" as str is deprecated. Use {new_target} instead',
                          category=DeprecationWarning)
            self.target = new_target
        # ---- Auto Backend
        if not self.backend:
            if self.target == Target.CPU:
                self.backend = Backend.C
            elif self.target == Target.GPU:
                self.backend = Backend.CUDA
            else:
                raise NotImplementedError(f'Target {self.target} has no default backend')

        #  Normalise data types
        if not isinstance(self.data_type, dict):
            dt = copy(self.data_type)  # The copy is necessary because BasicType has sympy shinanigans
            self.data_type = defaultdict(self.DataTypeFactory(dt))
        if not isinstance(self.default_number_float, BasicType):
            self.default_number_float = BasicType(self.default_number_float)
        if not isinstance(self.default_number_int, BasicType):
            self.default_number_int = BasicType(self.default_number_int)
