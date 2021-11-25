import itertools
import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Union, List, Dict, Tuple, Any

import sympy as sp

from pystencils.assignment import Assignment
from pystencils.astnodes import Block, Conditional, LoopOverCoordinate, SympyAssignment
from pystencils.cpu.vectorization import vectorize
from pystencils.enums import Target, Backend
from pystencils.field import Field, FieldType
from pystencils.gpucuda.indexing import indexing_creator_from_params
from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils.simp.simplifications import apply_sympy_optimisations
from pystencils.simplificationfactory import create_simplification_strategy
from pystencils.stencil import direction_string_to_offset, inverse_direction_string
from pystencils.transformations import (
    loop_blocking, move_constants_before_loop, remove_conditionals_in_staggered_kernel)


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
    data_type: Union[str, dict] = 'double'
    """
    Data type used for all untyped symbols (i.e. non-fields), can also be a dict from symbol name to type
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
    skip_independence_check: bool = False
    """
    Don't check that loop iterations are independent. This is needed e.g. for 
    periodicity kernel, that access the field outside the iteration bounds. Use with care!
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
    Either 'block' or 'line' , or custom indexing class, see `AbstractIndexing`
    """
    gpu_indexing_params: MappingProxyType = field(default=MappingProxyType({}))
    """
    Dict with indexing parameters (constructor parameters of indexing class)
    e.g. for 'block' one can specify '{'block_size': (20, 20, 10) }'.
    """
    default_assignment_simplifications: bool = False
    """
    If `True` default simplifications are first performed on the Assignments. If problems occur during the
    simplification a warning will be thrown. 
    Furthermore, it is essential to know that this is a two-stage process. The first stage of the process acts 
    on the level of the `AssignmentCollection`.  In this part, `create_simplification_strategy` 
    from pystencils.simplificationfactory will be used to apply optimisations like insertion of constants to 
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

    def __post_init__(self):
        # ----  Legacy parameters
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


def create_kernel(assignments: Union[Assignment, List[Assignment], AssignmentCollection, List[Conditional]], *,
                  config: CreateKernelConfig = None, **kwargs):
    """
    Creates abstract syntax tree (AST) of kernel, using a list of update equations.
    This function forms the general API and delegates the kernel creation to others depending on the CreateKernelConfig.
    Args:
        assignments: can be a single assignment, sequence of assignments or an `AssignmentCollection`
        config: CreateKernelConfig which includes the needed configuration
        kwargs: Arguments for updating the config

    Returns:
        abstract syntax tree (AST) object, that can either be printed as source code with `show_code` or
        can be compiled with through its 'compile()' member

    Example:
        >>> import pystencils as ps
        >>> import numpy as np
        >>> s, d = ps.fields('s, d: [2D]')
        >>> assignment = ps.Assignment(d[0,0], s[0, 1] + s[0, -1] + s[1, 0] + s[-1, 0])
        >>> kernel_ast = ps.create_kernel(assignment, config=ps.CreateKernelConfig(cpu_openmp=True))
        >>> kernel = kernel_ast.compile()
        >>> d_arr = np.zeros([5, 5])
        >>> kernel(d=d_arr, s=np.ones([5, 5]))
        >>> d_arr
        array([[0., 0., 0., 0., 0.],
               [0., 4., 4., 4., 0.],
               [0., 4., 4., 4., 0.],
               [0., 4., 4., 4., 0.],
               [0., 0., 0., 0., 0.]])
    """
    # ----  Updating configuration from kwargs
    if not config:
        config = CreateKernelConfig(**kwargs)
    else:
        for k, v in kwargs.items():
            if not hasattr(config, k):
                raise KeyError(f'{v} is not a valid kwarg. Please look in CreateKernelConfig for valid settings')
            setattr(config, k, v)

    # ----  Normalizing parameters
    if isinstance(assignments, Assignment):
        assignments = [assignments]
    assert assignments, "Assignments must not be empty!"

    if config.index_fields:
        return create_indexed_kernel(assignments, config=config)
    else:
        return create_domain_kernel(assignments, config=config)


def create_domain_kernel(assignments: List[Assignment], *, config: CreateKernelConfig):
    """
    Creates abstract syntax tree (AST) of kernel, using a list of update equations.

    Args:
        assignments: can be a single assignment, sequence of assignments or an `AssignmentCollection`
        config: CreateKernelConfig which includes the needed configuration

    Returns:
        abstract syntax tree (AST) object, that can either be printed as source code with `show_code` or
        can be compiled with through its 'compile()' member

    Example:
        >>> import pystencils as ps
        >>> import numpy as np
        >>> s, d = ps.fields('s, d: [2D]')
        >>> assignment = ps.Assignment(d[0,0], s[0, 1] + s[0, -1] + s[1, 0] + s[-1, 0])
        >>> kernel_config = ps.CreateKernelConfig(cpu_openmp=True)
        >>> kernel_ast = ps.kernelcreation.create_domain_kernel([assignment], config=kernel_config)
        >>> kernel = kernel_ast.compile()
        >>> d_arr = np.zeros([5, 5])
        >>> kernel(d=d_arr, s=np.ones([5, 5]))
        >>> d_arr
        array([[0., 0., 0., 0., 0.],
               [0., 4., 4., 4., 0.],
               [0., 4., 4., 4., 0.],
               [0., 4., 4., 4., 0.],
               [0., 0., 0., 0., 0.]])
    """
    # --- applying first default simplifications
    try:
        if config.default_assignment_simplifications and isinstance(assignments, AssignmentCollection):
            simplification = create_simplification_strategy()
            assignments = simplification(assignments)
    except Exception as e:
        warnings.warn(f"It was not possible to apply the default pystencils optimisations to the "
                      f"AssignmentCollection due to the following problem :{e}")

    # ----  Normalizing parameters
    split_groups = ()
    if isinstance(assignments, AssignmentCollection):
        if 'split_groups' in assignments.simplification_hints:
            split_groups = assignments.simplification_hints['split_groups']
        assignments = assignments.all_assignments

    try:
        if config.default_assignment_simplifications:
            assignments = apply_sympy_optimisations(assignments)
    except Exception as e:
        warnings.warn(f"It was not possible to apply the default SymPy optimisations to the "
                      f"Assignments due to the following problem :{e}")

    # ----  Creating ast
    ast = None
    if config.target == Target.CPU:
        if config.backend == Backend.C:
            from pystencils.cpu import add_openmp, create_kernel
            ast = create_kernel(assignments, function_name=config.function_name, type_info=config.data_type,
                                split_groups=split_groups,
                                iteration_slice=config.iteration_slice, ghost_layers=config.ghost_layers,
                                skip_independence_check=config.skip_independence_check)
            for optimization in config.cpu_prepend_optimizations:
                optimization(ast)
            omp_collapse = None
            if config.cpu_blocking:
                omp_collapse = loop_blocking(ast, config.cpu_blocking)
            if config.cpu_openmp:
                add_openmp(ast, num_threads=config.cpu_openmp, collapse=omp_collapse,
                           assume_single_outer_loop=config.omp_single_loop)
            if config.cpu_vectorize_info:
                if config.cpu_vectorize_info is True:
                    vectorize(ast)
                elif isinstance(config.cpu_vectorize_info, dict):
                    vectorize(ast, **config.cpu_vectorize_info)
                    if config.cpu_openmp and config.cpu_blocking and 'nontemporal' in config.cpu_vectorize_info and \
                            config.cpu_vectorize_info['nontemporal'] and 'cachelineZero' in ast.instruction_set:
                        # This condition is stricter than it needs to be: if blocks along the fastest axis start on a
                        # cache line boundary, it's okay. But we cannot determine that here.
                        # We don't need to disallow OpenMP collapsing because it is never applied to the inner loop.
                        raise ValueError("Blocking cannot be combined with cacheline-zeroing")
                else:
                    raise ValueError("Invalid value for cpu_vectorize_info")
    elif config.target == Target.GPU:
        if config.backend == Backend.CUDA:
            from pystencils.gpucuda import create_cuda_kernel
            ast = create_cuda_kernel(assignments, function_name=config.function_name, type_info=config.data_type,
                                     indexing_creator=indexing_creator_from_params(config.gpu_indexing,
                                                                                   config.gpu_indexing_params),
                                     iteration_slice=config.iteration_slice, ghost_layers=config.ghost_layers,
                                     skip_independence_check=config.skip_independence_check)

    if not ast:
        raise NotImplementedError(
            f'{config.target} together with {config.backend} is not supported by `create_domain_kernel`')

    if config.use_auto_for_assignments:
        for a in ast.atoms(SympyAssignment):
            a.use_auto = True

    return ast


def create_indexed_kernel(assignments: List[Assignment], *, config: CreateKernelConfig):
    """
    Similar to :func:`create_kernel`, but here not all cells of a field are updated but only cells with
    coordinates which are stored in an index field. This traversal method can e.g. be used for boundary handling.

    The coordinates are stored in a separated index_field, which is a one dimensional array with struct data type.
    This struct has to contain fields named 'x', 'y' and for 3D fields ('z'). These names are configurable with the
    'coordinate_names' parameter. The struct can have also other fields that can be read and written in the kernel, for
    example boundary parameters.

    Args:
        assignments: can be a single assignment, sequence of assignments or an `AssignmentCollection`
        config: CreateKernelConfig which includes the needed configuration

    Returns:
        abstract syntax tree (AST) object, that can either be printed as source code with `show_code` or
        can be compiled with through its 'compile()' member

    Example:
        >>> import pystencils as ps
        >>> import numpy as np
        >>>
        >>> # Index field stores the indices of the cell to visit together with optional values
        >>> index_arr_dtype = np.dtype([('x', np.int32), ('y', np.int32), ('val', np.double)])
        >>> index_arr = np.array([(1, 1, 0.1), (2, 2, 0.2), (3, 3, 0.3)], dtype=index_arr_dtype)
        >>> idx_field = ps.fields(idx=index_arr)
        >>>
        >>> # Additional values  stored in index field can be accessed in the kernel as well
        >>> s, d = ps.fields('s, d: [2D]')
        >>> assignment = ps.Assignment(d[0, 0], 2 * s[0, 1] + 2 * s[1, 0] + idx_field('val'))
        >>> kernel_config = ps.CreateKernelConfig(index_fields=[idx_field], coordinate_names=('x', 'y'))
        >>> kernel_ast = ps.create_indexed_kernel([assignment], config=kernel_config)
        >>> kernel = kernel_ast.compile()
        >>> d_arr = np.zeros([5, 5])
        >>> kernel(s=np.ones([5, 5]), d=d_arr, idx=index_arr)
        >>> d_arr
        array([[0. , 0. , 0. , 0. , 0. ],
               [0. , 4.1, 0. , 0. , 0. ],
               [0. , 0. , 4.2, 0. , 0. ],
               [0. , 0. , 0. , 4.3, 0. ],
               [0. , 0. , 0. , 0. , 0. ]])
    """
    ast = None
    if config.target == Target.CPU and config.backend == Backend.C:
        from pystencils.cpu import add_openmp, create_indexed_kernel
        ast = create_indexed_kernel(assignments, index_fields=config.index_fields, type_info=config.data_type,
                                    coordinate_names=config.coordinate_names)
        if config.cpu_openmp:
            add_openmp(ast, num_threads=config.cpu_openmp)
    elif config.target == Target.GPU:
        if config.backend == Backend.CUDA:
            from pystencils.gpucuda import created_indexed_cuda_kernel
            idx_creator = indexing_creator_from_params(config.gpu_indexing, config.gpu_indexing_params)
            ast = created_indexed_cuda_kernel(assignments,
                                              config.index_fields,
                                              type_info=config.data_type,
                                              coordinate_names=config.coordinate_names,
                                              indexing_creator=idx_creator)

    if not ast:
        raise NotImplementedError(f'Indexed kernels are not yet supported for {config.target} with {config.backend}')
    return ast


def create_staggered_kernel(assignments, target: Target = Target.CPU, gpu_exclusive_conditions=False, **kwargs):
    """Kernel that updates a staggered field.

    .. image:: /img/staggered_grid.svg

    For a staggered field, the first index coordinate defines the location of the staggered value.
    Further index coordinates can be used to store vectors/tensors at each point.

    Args:
        assignments: a sequence of assignments or an AssignmentCollection.
                     Assignments to staggered field are processed specially, while subexpressions and assignments to
                     regular fields are passed through to `create_kernel`. Multiple different staggered fields can be
                     used, but they all need to use the same stencil (i.e. the same number of staggered points) and
                     shape.
        target: 'CPU' or 'GPU'
        gpu_exclusive_conditions: disable the use of multiple conditionals inside the loop. The outer layers are then
                                  handled in an else branch.
        kwargs: passed directly to create_kernel, iteration_slice and ghost_layers parameters are not allowed

    Returns:
        AST, see `create_kernel`
    """
    if 'ghost_layers' in kwargs:
        assert kwargs['ghost_layers'] is None
        del kwargs['ghost_layers']
    if 'iteration_slice' in kwargs:
        assert kwargs['iteration_slice'] is None
        del kwargs['iteration_slice']
    if 'omp_single_loop' in kwargs:
        assert kwargs['omp_single_loop'] is False
        del kwargs['omp_single_loop']

    if isinstance(assignments, AssignmentCollection):
        subexpressions = assignments.subexpressions + [a for a in assignments.main_assignments
                                                       if not hasattr(a, 'lhs')
                                                       or type(a.lhs) is not Field.Access
                                                       or not FieldType.is_staggered(a.lhs.field)]
        assignments = [a for a in assignments.main_assignments if hasattr(a, 'lhs')
                       and type(a.lhs) is Field.Access
                       and FieldType.is_staggered(a.lhs.field)]
    else:
        subexpressions = [a for a in assignments if not hasattr(a, 'lhs')
                          or type(a.lhs) is not Field.Access
                          or not FieldType.is_staggered(a.lhs.field)]
        assignments = [a for a in assignments if hasattr(a, 'lhs')
                       and type(a.lhs) is Field.Access
                       and FieldType.is_staggered(a.lhs.field)]
    if len(set([tuple(a.lhs.field.staggered_stencil) for a in assignments])) != 1:
        raise ValueError("All assignments need to be made to staggered fields with the same stencil")
    if len(set([a.lhs.field.shape for a in assignments])) != 1:
        raise ValueError("All assignments need to be made to staggered fields with the same shape")

    staggered_field = assignments[0].lhs.field
    stencil = staggered_field.staggered_stencil
    dim = staggered_field.spatial_dimensions
    shape = staggered_field.shape

    counters = [LoopOverCoordinate.get_loop_counter_symbol(i) for i in range(dim)]

    final_assignments = []

    # find out whether any of the ghost layers is not needed
    common_exclusions = set(["E", "W", "N", "S", "T", "B"][:2 * dim])
    for direction in stencil:
        exclusions = set(["E", "W", "N", "S", "T", "B"][:2 * dim])
        for elementary_direction in direction:
            exclusions.remove(inverse_direction_string(elementary_direction))
        common_exclusions.intersection_update(exclusions)
    ghost_layers = [[0, 0] for d in range(dim)]
    for direction in common_exclusions:
        direction = direction_string_to_offset(direction)
        for d, s in enumerate(direction):
            if s == 1:
                ghost_layers[d][1] = 1
            elif s == -1:
                ghost_layers[d][0] = 1

    def condition(direction):
        """exclude those staggered points that correspond to fluxes between ghost cells"""
        exclusions = set(["E", "W", "N", "S", "T", "B"][:2 * dim])

        for elementary_direction in direction:
            exclusions.remove(inverse_direction_string(elementary_direction))
        conditions = []
        for e in exclusions:
            if e in common_exclusions:
                continue
            offset = direction_string_to_offset(e)
            for i, o in enumerate(offset):
                if o == 1:
                    conditions.append(counters[i] < shape[i] - 1)
                elif o == -1:
                    conditions.append(counters[i] > 0)
        return sp.And(*conditions)

    if gpu_exclusive_conditions:
        outer_assignment = None
        conditions = {direction: condition(direction) for direction in stencil}
        for num_conditions in range(len(stencil)):
            for combination in itertools.combinations(conditions.values(), num_conditions):
                for assignment in assignments:
                    direction = stencil[assignment.lhs.index[0]]
                    if conditions[direction] in combination:
                        assignment = SympyAssignment(assignment.lhs, assignment.rhs)
                        outer_assignment = Conditional(sp.And(*combination), Block([assignment]), outer_assignment)

        inner_assignment = []
        for assignment in assignments:
            inner_assignment.append(SympyAssignment(assignment.lhs, assignment.rhs))
        last_conditional = Conditional(sp.And(*[condition(d) for d in stencil]),
                                       Block(inner_assignment), outer_assignment)
        final_assignments = [s for s in subexpressions if not hasattr(s, 'lhs')] + \
                            [SympyAssignment(s.lhs, s.rhs) for s in subexpressions if hasattr(s, 'lhs')] + \
                            [last_conditional]

        if target == Target.CPU:
            from pystencils.cpu import create_kernel as create_kernel_cpu
            ast = create_kernel_cpu(final_assignments, ghost_layers=ghost_layers, omp_single_loop=False, **kwargs)
        else:
            ast = create_kernel(final_assignments, ghost_layers=ghost_layers, target=target, **kwargs)
        return ast

    for assignment in assignments:
        direction = stencil[assignment.lhs.index[0]]
        sp_assignments = [s for s in subexpressions if not hasattr(s, 'lhs')] + \
                         [SympyAssignment(s.lhs, s.rhs) for s in subexpressions if hasattr(s, 'lhs')] + \
                         [SympyAssignment(assignment.lhs, assignment.rhs)]
        last_conditional = Conditional(condition(direction), Block(sp_assignments))
        final_assignments.append(last_conditional)

    remove_start_conditional = any([gl[0] == 0 for gl in ghost_layers])
    prepend_optimizations = [lambda ast: remove_conditionals_in_staggered_kernel(ast, remove_start_conditional),
                             move_constants_before_loop]
    if 'cpu_prepend_optimizations' in kwargs:
        prepend_optimizations += kwargs['cpu_prepend_optimizations']
        del kwargs['cpu_prepend_optimizations']
    ast = create_kernel(final_assignments, ghost_layers=ghost_layers, target=target, omp_single_loop=False,
                        cpu_prepend_optimizations=prepend_optimizations, **kwargs)
    return ast
