import functools
import itertools
from types import MappingProxyType

import sympy as sp

from pystencils.assignment import Assignment
from pystencils.astnodes import Block, Conditional, LoopOverCoordinate, SympyAssignment
from pystencils.cpu.vectorization import vectorize
from pystencils.field import Field, FieldType
from pystencils.gpucuda.indexing import indexing_creator_from_params
from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils.stencil import direction_string_to_offset, inverse_direction_string
from pystencils.transformations import (
    loop_blocking, move_constants_before_loop, remove_conditionals_in_staggered_kernel)


def create_kernel(assignments,
                  target='cpu',
                  data_type="double",
                  iteration_slice=None,
                  ghost_layers=None,
                  skip_independence_check=False,
                  cpu_openmp=False,
                  cpu_vectorize_info=None,
                  cpu_blocking=None,
                  omp_single_loop=True,
                  gpu_indexing='block',
                  gpu_indexing_params=MappingProxyType({}),
                  use_textures_for_interpolation=True,
                  cpu_prepend_optimizations=[],
                  use_auto_for_assignments=False,
                  opencl_queue=None,
                  opencl_ctx=None):
    """
    Creates abstract syntax tree (AST) of kernel, using a list of update equations.

    Args:
        assignments: can be a single assignment, sequence of assignments or an `AssignmentCollection`
        target: 'cpu', 'llvm', 'gpu' or 'opencl'
        data_type: data type used for all untyped symbols (i.e. non-fields), can also be a dict from symbol name
                  to type
        iteration_slice: rectangular subset to iterate over, if not specified the complete non-ghost layer \
                         part of the field is iterated over
        ghost_layers: a single integer specifies the ghost layer count at all borders, can also be a sequence of
                      pairs ``[(x_lower_gl, x_upper_gl), .... ]``. These layers are excluded from the iteration.
                      If left to default, the number of ghost layers is determined automatically.
        skip_independence_check: don't check that loop iterations are independent. This is needed e.g. for
                                 periodicity kernel, that access the field outside the iteration bounds. Use with care!
        cpu_openmp: True or number of threads for OpenMP parallelization, False for no OpenMP
        omp_single_loop: if OpenMP is active: whether multiple outer loops are permitted
        cpu_vectorize_info: a dictionary with keys, 'vector_instruction_set', 'assume_aligned' and 'nontemporal'
                            for documentation of these parameters see vectorize function. Example:
                            '{'instruction_set': 'avx512', 'assume_aligned': True, 'nontemporal':True}'
        cpu_blocking: a tuple of block sizes or None if no blocking should be applied
        gpu_indexing: either 'block' or 'line' , or custom indexing class, see `AbstractIndexing`
        gpu_indexing_params: dict with indexing parameters (constructor parameters of indexing class)
                             e.g. for 'block' one can specify '{'block_size': (20, 20, 10) }'
        cpu_prepend_optimizations: list of extra optimizations to perform first on the AST

    Returns:
        abstract syntax tree (AST) object, that can either be printed as source code with `show_code` or
        can be compiled with through its 'compile()' member

    Example:
        >>> import pystencils as ps
        >>> import numpy as np
        >>> s, d = ps.fields('s, d: [2D]')
        >>> assignment = ps.Assignment(d[0,0], s[0, 1] + s[0, -1] + s[1, 0] + s[-1, 0])
        >>> ast = ps.create_kernel(assignment, target='cpu', cpu_openmp=True)
        >>> kernel = ast.compile()
        >>> d_arr = np.zeros([5, 5])
        >>> kernel(d=d_arr, s=np.ones([5, 5]))
        >>> d_arr
        array([[0., 0., 0., 0., 0.],
               [0., 4., 4., 4., 0.],
               [0., 4., 4., 4., 0.],
               [0., 4., 4., 4., 0.],
               [0., 0., 0., 0., 0.]])
    """
    # ----  Normalizing parameters
    if isinstance(assignments, Assignment):
        assignments = [assignments]
    assert assignments, "Assignments must not be empty!"
    split_groups = ()
    if isinstance(assignments, AssignmentCollection):
        if 'split_groups' in assignments.simplification_hints:
            split_groups = assignments.simplification_hints['split_groups']
        assignments = assignments.all_assignments

    # ----  Creating ast
    if target == 'cpu':
        from pystencils.cpu import create_kernel
        from pystencils.cpu import add_openmp
        ast = create_kernel(assignments, type_info=data_type, split_groups=split_groups,
                            iteration_slice=iteration_slice, ghost_layers=ghost_layers,
                            skip_independence_check=skip_independence_check)
        for optimization in cpu_prepend_optimizations:
            optimization(ast)
        omp_collapse = None
        if cpu_blocking:
            omp_collapse = loop_blocking(ast, cpu_blocking)
        if cpu_openmp:
            add_openmp(ast, num_threads=cpu_openmp, collapse=omp_collapse, assume_single_outer_loop=omp_single_loop)
        if cpu_vectorize_info:
            if cpu_vectorize_info is True:
                vectorize(ast)
            elif isinstance(cpu_vectorize_info, dict):
                vectorize(ast, **cpu_vectorize_info)
                if cpu_openmp and cpu_blocking and 'nontemporal' in cpu_vectorize_info and \
                        cpu_vectorize_info['nontemporal'] and 'cachelineZero' in ast.instruction_set:
                    # This condition is stricter than it needs to be: if blocks along the fastest axis start on a
                    # cache line boundary, it's okay. But we cannot determine that here.
                    # We don't need to disallow OpenMP collapsing because it is never applied to the inner loop.
                    raise ValueError("Blocking cannot be combined with cacheline-zeroing")
            else:
                raise ValueError("Invalid value for cpu_vectorize_info")
    elif target == 'llvm':
        from pystencils.llvm import create_kernel
        ast = create_kernel(assignments, type_info=data_type, split_groups=split_groups,
                            iteration_slice=iteration_slice, ghost_layers=ghost_layers)
    elif target == 'gpu' or target == 'opencl':
        from pystencils.gpucuda import create_cuda_kernel
        ast = create_cuda_kernel(assignments, type_info=data_type,
                                 indexing_creator=indexing_creator_from_params(gpu_indexing, gpu_indexing_params),
                                 iteration_slice=iteration_slice, ghost_layers=ghost_layers,
                                 skip_independence_check=skip_independence_check,
                                 use_textures_for_interpolation=use_textures_for_interpolation)
        if target == 'opencl':
            from pystencils.opencl.opencljit import make_python_function
            ast._backend = 'opencl'
            ast.compile = functools.partial(make_python_function, ast, opencl_queue, opencl_ctx)
            ast._target = 'opencl'
            ast._backend = 'opencl'
        return ast
    else:
        raise ValueError(f"Unknown target {target}. Has to be one of 'cpu', 'gpu' or 'llvm' ")

    if use_auto_for_assignments:
        for a in ast.atoms(SympyAssignment):
            a.use_auto = True

    return ast


def create_indexed_kernel(assignments,
                          index_fields,
                          target='cpu',
                          data_type="double",
                          coordinate_names=('x', 'y', 'z'),
                          cpu_openmp=True,
                          gpu_indexing='block',
                          gpu_indexing_params=MappingProxyType({}),
                          use_textures_for_interpolation=True,
                          opencl_queue=None,
                          opencl_ctx=None):
    """
    Similar to :func:`create_kernel`, but here not all cells of a field are updated but only cells with
    coordinates which are stored in an index field. This traversal method can e.g. be used for boundary handling.

    The coordinates are stored in a separated index_field, which is a one dimensional array with struct data type.
    This struct has to contain fields named 'x', 'y' and for 3D fields ('z'). These names are configurable with the
    'coordinate_names' parameter. The struct can have also other fields that can be read and written in the kernel, for
    example boundary parameters.

    index_fields: list of index fields, i.e. 1D fields with struct data type
    coordinate_names: name of the coordinate fields in the struct data type

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
        >>> assignment = ps.Assignment(d[0,0], 2 * s[0, 1] + 2 * s[1, 0] + idx_field('val'))
        >>> ast = create_indexed_kernel(assignment, [idx_field], coordinate_names=('x', 'y'))
        >>> kernel = ast.compile()
        >>> d_arr = np.zeros([5, 5])
        >>> kernel(s=np.ones([5, 5]), d=d_arr, idx=index_arr)
        >>> d_arr
        array([[0. , 0. , 0. , 0. , 0. ],
               [0. , 4.1, 0. , 0. , 0. ],
               [0. , 0. , 4.2, 0. , 0. ],
               [0. , 0. , 0. , 4.3, 0. ],
               [0. , 0. , 0. , 0. , 0. ]])
    """
    if isinstance(assignments, Assignment):
        assignments = [assignments]
    elif isinstance(assignments, AssignmentCollection):
        assignments = assignments.all_assignments
    if target == 'cpu':
        from pystencils.cpu import create_indexed_kernel
        from pystencils.cpu import add_openmp
        ast = create_indexed_kernel(assignments, index_fields=index_fields, type_info=data_type,
                                    coordinate_names=coordinate_names)
        if cpu_openmp:
            add_openmp(ast, num_threads=cpu_openmp)
        return ast
    elif target == 'llvm':
        raise NotImplementedError("Indexed kernels are not yet supported in LLVM backend")
    elif target == 'gpu' or target == 'opencl':
        from pystencils.gpucuda import created_indexed_cuda_kernel
        idx_creator = indexing_creator_from_params(gpu_indexing, gpu_indexing_params)
        ast = created_indexed_cuda_kernel(assignments,
                                          index_fields,
                                          type_info=data_type,
                                          coordinate_names=coordinate_names,
                                          indexing_creator=idx_creator,
                                          use_textures_for_interpolation=use_textures_for_interpolation)
        if target == 'opencl':
            from pystencils.opencl.opencljit import make_python_function
            ast._backend = 'opencl'
            ast.compile = functools.partial(make_python_function, ast, opencl_queue, opencl_ctx)
            ast._target = 'opencl'
            ast._backend = 'opencl'
        return ast
    else:
        raise ValueError(f"Unknown target {target}. Has to be either 'cpu' or 'gpu'")


def create_staggered_kernel(assignments, target='cpu', gpu_exclusive_conditions=False, **kwargs):
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
        target: 'cpu', 'llvm' or 'gpu'
        gpu_exclusive_conditions: disable the use of multiple conditionals inside the loop. The outer layers are then
                                  handled in an else branch.
        kwargs: passed directly to create_kernel, iteration_slice and ghost_layers parameters are not allowed

    Returns:
        AST, see `create_kernel`
    """
    assert 'iteration_slice' not in kwargs and 'ghost_layers' not in kwargs and 'omp_single_loop' not in kwargs

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
            direction = stencil[assignment.lhs.index[0]]
            inner_assignment.append(SympyAssignment(assignment.lhs, assignment.rhs))
        last_conditional = Conditional(sp.And(*[condition(d) for d in stencil]),
                                       Block(inner_assignment), outer_assignment)
        final_assignments = [s for s in subexpressions if not hasattr(s, 'lhs')] + \
                            [SympyAssignment(s.lhs, s.rhs) for s in subexpressions if hasattr(s, 'lhs')] + \
                            [last_conditional]

        if target == 'cpu':
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
    ast = create_kernel(final_assignments, ghost_layers=ghost_layers, target=target, omp_single_loop=False,
                        cpu_prepend_optimizations=prepend_optimizations, **kwargs)
    return ast
