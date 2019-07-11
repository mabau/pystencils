import itertools
from types import MappingProxyType

import sympy as sp

from pystencils.assignment import Assignment
from pystencils.astnodes import Block, Conditional, LoopOverCoordinate, SympyAssignment
from pystencils.cpu.vectorization import vectorize
from pystencils.gpucuda.indexing import indexing_creator_from_params
from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils.transformations import (
    loop_blocking, move_constants_before_loop, remove_conditionals_in_staggered_kernel)


def create_kernel(assignments, target='cpu', data_type="double", iteration_slice=None, ghost_layers=None,
                  skip_independence_check=False,
                  cpu_openmp=False, cpu_vectorize_info=None, cpu_blocking=None,
                  gpu_indexing='block', gpu_indexing_params=MappingProxyType({})):
    """
    Creates abstract syntax tree (AST) of kernel, using a list of update equations.

    Args:
        assignments: can be a single assignment, sequence of assignments or an `AssignmentCollection`
        target: 'cpu', 'llvm' or 'gpu'
        data_type: data type used for all untyped symbols (i.e. non-fields), can also be a dict from symbol name
                  to type
        iteration_slice: rectangular subset to iterate over, if not specified the complete non-ghost layer \
                         part of the field is iterated over
        ghost_layers: if left to default, the number of necessary ghost layers is determined automatically
                     a single integer specifies the ghost layer count at all borders, can also be a sequence of
                     pairs ``[(x_lower_gl, x_upper_gl), .... ]``
        skip_independence_check: don't check that loop iterations are independent. This is needed e.g. for
                                 periodicity kernel, that access the field outside the iteration bounds. Use with care!
        cpu_openmp: True or number of threads for OpenMP parallelization, False for no OpenMP
        cpu_vectorize_info: a dictionary with keys, 'vector_instruction_set', 'assume_aligned' and 'nontemporal'
                            for documentation of these parameters see vectorize function. Example:
                            '{'instruction_set': 'avx512', 'assume_aligned': True, 'nontemporal':True}'
        cpu_blocking: a tuple of block sizes or None if no blocking should be applied
        gpu_indexing: either 'block' or 'line' , or custom indexing class, see `AbstractIndexing`
        gpu_indexing_params: dict with indexing parameters (constructor parameters of indexing class)
                             e.g. for 'block' one can specify '{'block_size': (20, 20, 10) }'

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
    split_groups = ()
    if isinstance(assignments, AssignmentCollection):
        if 'split_groups' in assignments.simplification_hints:
            split_groups = assignments.simplification_hints['split_groups']
        assignments = assignments.all_assignments
    if isinstance(assignments, Assignment):
        assignments = [assignments]

    # ----  Creating ast
    if target == 'cpu':
        from pystencils.cpu import create_kernel
        from pystencils.cpu import add_openmp
        ast = create_kernel(assignments, type_info=data_type, split_groups=split_groups,
                            iteration_slice=iteration_slice, ghost_layers=ghost_layers,
                            skip_independence_check=skip_independence_check)
        omp_collapse = None
        if cpu_blocking:
            omp_collapse = loop_blocking(ast, cpu_blocking)
        if cpu_openmp:
            add_openmp(ast, num_threads=cpu_openmp, collapse=omp_collapse)
        if cpu_vectorize_info:
            if cpu_vectorize_info is True:
                vectorize(ast)
            elif isinstance(cpu_vectorize_info, dict):
                vectorize(ast, **cpu_vectorize_info)
            else:
                raise ValueError("Invalid value for cpu_vectorize_info")
        return ast
    elif target == 'llvm':
        from pystencils.llvm import create_kernel
        ast = create_kernel(assignments, type_info=data_type, split_groups=split_groups,
                            iteration_slice=iteration_slice, ghost_layers=ghost_layers)
        return ast
    elif target == 'gpu':
        from pystencils.gpucuda import create_cuda_kernel
        ast = create_cuda_kernel(assignments, type_info=data_type,
                                 indexing_creator=indexing_creator_from_params(gpu_indexing, gpu_indexing_params),
                                 iteration_slice=iteration_slice, ghost_layers=ghost_layers,
                                 skip_independence_check=skip_independence_check)
        return ast
    else:
        raise ValueError("Unknown target %s. Has to be one of 'cpu', 'gpu' or 'llvm' " % (target,))


def create_indexed_kernel(assignments, index_fields, target='cpu', data_type="double", coordinate_names=('x', 'y', 'z'),
                          cpu_openmp=True, gpu_indexing='block', gpu_indexing_params=MappingProxyType({})):
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
    elif target == 'gpu':
        from pystencils.gpucuda import created_indexed_cuda_kernel
        idx_creator = indexing_creator_from_params(gpu_indexing, gpu_indexing_params)
        ast = created_indexed_cuda_kernel(assignments, index_fields, type_info=data_type,
                                          coordinate_names=coordinate_names, indexing_creator=idx_creator)
        return ast
    else:
        raise ValueError("Unknown target %s. Has to be either 'cpu' or 'gpu'" % (target,))


def create_staggered_kernel(staggered_field, expressions, subexpressions=(), target='cpu',
                            gpu_exclusive_conditions=False, **kwargs):
    """Kernel that updates a staggered field.

    .. image:: /img/staggered_grid.svg

    Args:
        staggered_field: field where the first index coordinate defines the location of the staggered value
                can have 1 or 2 index coordinates, in case of two index coordinates at every staggered location
                a vector is stored, expressions parameter has to be a sequence of sequences then
                where e.g. ``f[0,0](0)`` is interpreted as value at the left cell boundary, ``f[1,0](0)`` the right cell
                boundary and ``f[0,0](1)`` the southern cell boundary etc.
        expressions: sequence of expressions of length dim, defining how the west, southern, (bottom) cell boundary
                     should be updated.
        subexpressions: optional sequence of Assignments, that define subexpressions used in the main expressions
        target: 'cpu' or 'gpu'
        gpu_exclusive_conditions: if/else construct to have only one code block for each of 2**dim code paths
        kwargs: passed directly to create_kernel, iteration slice and ghost_layers parameters are not allowed

    Returns:
        AST, see `create_kernel`
    """
    assert 'iteration_slice' not in kwargs and 'ghost_layers' not in kwargs
    assert staggered_field.index_dimensions in (1, 2), 'Staggered field must have one or two index dimensions'
    dim = staggered_field.spatial_dimensions

    counters = [LoopOverCoordinate.get_loop_counter_symbol(i) for i in range(dim)]
    conditions = [counters[i] < staggered_field.shape[i] - 1 for i in range(dim)]
    assert len(expressions) == dim
    if staggered_field.index_dimensions == 2:
        assert all(len(sublist) == len(expressions[0]) for sublist in expressions), \
            "If staggered field has two index dimensions expressions has to be a sequence of sequences of all the " \
            "same length."

    final_assignments = []
    last_conditional = None

    def add(condition, dimensions, as_else_block=False):
        nonlocal last_conditional
        if staggered_field.index_dimensions == 1:
            assignments = [Assignment(staggered_field(d), expressions[d]) for d in dimensions]
            a_coll = AssignmentCollection(assignments, list(subexpressions))
            a_coll = a_coll.new_filtered([staggered_field(d) for d in dimensions])
        elif staggered_field.index_dimensions == 2:
            assert staggered_field.has_fixed_index_shape
            assignments = [Assignment(staggered_field(d, i), expr)
                           for d in dimensions
                           for i, expr in enumerate(expressions[d])]
            a_coll = AssignmentCollection(assignments, list(subexpressions))
            a_coll = a_coll.new_filtered([staggered_field(d, i) for i in range(staggered_field.index_shape[1])
                                          for d in dimensions])
        sp_assignments = [SympyAssignment(a.lhs, a.rhs) for a in a_coll.all_assignments]
        if as_else_block and last_conditional:
            new_cond = Conditional(condition, Block(sp_assignments))
            last_conditional.false_block = Block([new_cond])
            last_conditional = new_cond
        else:
            last_conditional = Conditional(condition, Block(sp_assignments))
            final_assignments.append(last_conditional)

    if target == 'cpu' or not gpu_exclusive_conditions:
        for d in range(dim):
            cond = sp.And(*[conditions[i] for i in range(dim) if d != i])
            add(cond, [d])
    elif target == 'gpu':
        full_conditions = [sp.And(*[conditions[i] for i in range(dim) if d != i]) for d in range(dim)]
        for include in itertools.product(*[[1, 0]] * dim):
            case_conditions = sp.And(*[c if value else sp.Not(c) for c, value in zip(full_conditions, include)])
            dimensions_to_include = [i for i in range(dim) if include[i]]
            if dimensions_to_include:
                add(case_conditions, dimensions_to_include, True)

    ghost_layers = [(1, 0)] * dim

    blocking = kwargs.get('cpu_blocking', None)
    if blocking:
        del kwargs['cpu_blocking']

    cpu_vectorize_info = kwargs.get('cpu_vectorize_info', None)
    if cpu_vectorize_info:
        del kwargs['cpu_vectorize_info']
    openmp = kwargs.get('cpu_openmp', None)
    if openmp:
        del kwargs['cpu_openmp']

    ast = create_kernel(final_assignments, ghost_layers=ghost_layers, target=target, **kwargs)

    if target == 'cpu':
        remove_conditionals_in_staggered_kernel(ast)
        move_constants_before_loop(ast)
        omp_collapse = None
        if blocking:
            omp_collapse = loop_blocking(ast, blocking)
        if openmp:
            from pystencils.cpu import add_openmp
            add_openmp(ast, num_threads=openmp, collapse=omp_collapse, assume_single_outer_loop=False)
        if cpu_vectorize_info is True:
            vectorize(ast)
        elif isinstance(cpu_vectorize_info, dict):
            vectorize(ast, **cpu_vectorize_info)
    return ast
