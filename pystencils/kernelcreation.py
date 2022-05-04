import itertools
import warnings
from typing import Union, List

import sympy as sp
from pystencils.config import CreateKernelConfig

from pystencils.assignment import Assignment
from pystencils.astnodes import Node, Block, Conditional, LoopOverCoordinate, SympyAssignment
from pystencils.cpu.vectorization import vectorize
from pystencils.enums import Target, Backend
from pystencils.field import Field, FieldType
from pystencils.node_collection import NodeCollection
from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils.kernel_contrains_check import KernelConstraintsCheck
from pystencils.simplificationfactory import create_simplification_strategy
from pystencils.stencil import direction_string_to_offset, inverse_direction_string
from pystencils.transformations import (
    loop_blocking, move_constants_before_loop, remove_conditionals_in_staggered_kernel)


def create_kernel(assignments: Union[Assignment, List[Assignment], AssignmentCollection, List[Node], NodeCollection], *,
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
    if isinstance(assignments, list):
        assignments = NodeCollection(assignments)
    elif isinstance(assignments, AssignmentCollection):
        # TODO Markus check and doku
        # --- applying first default simplifications
        try:
            if config.default_assignment_simplifications:
                simplification = create_simplification_strategy()
                assignments = simplification(assignments)
        except Exception as e:
            warnings.warn(f"It was not possible to apply the default pystencils optimisations to the "
                          f"AssignmentCollection due to the following problem :{e}")
        simplification_hints = assignments.simplification_hints
        assignments = NodeCollection.from_assignment_collection(assignments)
        assignments.simplification_hints = simplification_hints

    if config.index_fields:
        return create_indexed_kernel(assignments, config=config)
    else:
        return create_domain_kernel(assignments, config=config)


def create_domain_kernel(assignments: NodeCollection, *, config: CreateKernelConfig):
    """
    Creates abstract syntax tree (AST) of kernel, using a list of update equations.

    Note that `create_domain_kernel` is a lower level function which shoul be accessed by not providing `index_fields`
    to create_kernel

    Args:
        assignments: can be a single assignment, sequence of assignments or an `AssignmentCollection`
        config: CreateKernelConfig which includes the needed configuration

    Returns:
        abstract syntax tree (AST) object, that can either be printed as source code with `show_code` or
        can be compiled with through its 'compile()' member

    Example:
        >>> import pystencils as ps
        >>> import numpy as np
        >>> from pystencils.kernelcreation import create_domain_kernel
        >>> from pystencils.node_collection import NodeCollection
        >>> s, d = ps.fields('s, d: [2D]')
        >>> assignment = ps.Assignment(d[0,0], s[0, 1] + s[0, -1] + s[1, 0] + s[-1, 0])
        >>> kernel_config = ps.CreateKernelConfig(cpu_openmp=True)
        >>> kernel_ast = create_domain_kernel(NodeCollection([assignment]), config=kernel_config)
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
    # --- eval
    assignments.evaluate_terms()

    # FUTURE WORK from here we shouldn't NEED sympy
    # --- check constrains
    check = KernelConstraintsCheck(check_independence_condition=not config.skip_independence_check,
                                   check_double_write_condition=not config.allow_double_writes)
    check.visit(assignments)

    assignments.bound_fields = check.fields_written
    assignments.rhs_fields = check.fields_read

    # ----  Creating ast
    ast = None
    if config.target == Target.CPU:
        if config.backend == Backend.C:
            from pystencils.cpu import add_openmp, create_kernel
            ast = create_kernel(assignments, config=config)
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
            ast = create_cuda_kernel(assignments, config=config)

    if not ast:
        raise NotImplementedError(
            f'{config.target} together with {config.backend} is not supported by `create_domain_kernel`')

    if config.use_auto_for_assignments:
        for a in ast.atoms(SympyAssignment):
            a.use_auto = True

    return ast


def create_indexed_kernel(assignments: NodeCollection, *, config: CreateKernelConfig):
    """
    Similar to :func:`create_kernel`, but here not all cells of a field are updated but only cells with
    coordinates which are stored in an index field. This traversal method can e.g. be used for boundary handling.

    The coordinates are stored in a separated index_field, which is a one dimensional array with struct data type.
    This struct has to contain fields named 'x', 'y' and for 3D fields ('z'). These names are configurable with the
    'coordinate_names' parameter. The struct can have also other fields that can be read and written in the kernel, for
    example boundary parameters.

    Note that `create_indexed_kernel` is a lower level function which shoul be accessed by providing `index_fields`
    to create_kernel

    Args:
        assignments: can be a single assignment, sequence of assignments or an `AssignmentCollection`
        config: CreateKernelConfig which includes the needed configuration

    Returns:
        abstract syntax tree (AST) object, that can either be printed as source code with `show_code` or
        can be compiled with through its 'compile()' member

    Example:
        >>> import pystencils as ps
        >>> from pystencils.node_collection import NodeCollection
        >>> import numpy as np
        >>> from pystencils.kernelcreation import create_indexed_kernel
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
        >>> kernel_ast = create_indexed_kernel(NodeCollection([assignment]), config=kernel_config)
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
    # --- eval
    assignments.evaluate_terms()

    # FUTURE WORK from here we shouldn't NEED sympy
    # --- check constrains
    check = KernelConstraintsCheck(check_independence_condition=not config.skip_independence_check,
                                   check_double_write_condition=not config.allow_double_writes)
    check.visit(assignments)

    assignments.bound_fields = check.fields_written
    assignments.rhs_fields = check.fields_read

    ast = None
    if config.target == Target.CPU and config.backend == Backend.C:
        from pystencils.cpu import add_openmp, create_indexed_kernel
        ast = create_indexed_kernel(assignments, config=config)
        if config.cpu_openmp:
            add_openmp(ast, num_threads=config.cpu_openmp)
    elif config.target == Target.GPU:
        if config.backend == Backend.CUDA:
            from pystencils.gpucuda import created_indexed_cuda_kernel
            ast = created_indexed_cuda_kernel(assignments, config=config)

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
    # TODO: Add doku like in the other kernels
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

        config = CreateKernelConfig(target=target, ghost_layers=ghost_layers, omp_single_loop=False, **kwargs)
        ast = create_kernel(final_assignments, config=config)
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

    config = CreateKernelConfig(ghost_layers=ghost_layers, target=target, omp_single_loop=False,
                                cpu_prepend_optimizations=prepend_optimizations, **kwargs)
    ast = create_kernel(final_assignments, config=config)
    return ast
