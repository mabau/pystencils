from pystencils.assignment_collection import AssignmentCollection
from pystencils.gpucuda.indexing import indexing_creator_from_params


def create_kernel(equations, target='cpu', data_type="double", iteration_slice=None, ghost_layers=None,
                  cpu_openmp=False, cpu_vectorize_info=None,
                  gpu_indexing='block', gpu_indexing_params={}):
    """
    Creates abstract syntax tree (AST) of kernel, using a list of update equations.
    :param equations: either be a plain list of equations or a AssignmentCollection object
    :param target: 'cpu', 'llvm' or 'gpu'
    :param data_type: data type used for all untyped symbols (i.e. non-fields), can also be a dict from symbol name
                     to type
    :param iteration_slice: rectangular subset to iterate over, if not specified the complete non-ghost layer part of the
                           field is iterated over
    :param ghost_layers: if left to default, the number of necessary ghost layers is determined automatically
                        a single integer specifies the ghost layer count at all borders, can also be a sequence of
                        pairs [(xLowerGl, xUpperGl), .... ]

    CPU specific Parameters:
    :param cpu_openmp: True or number of threads for OpenMP parallelization, False for no OpenMP
    :param cpu_vectorize_info: pair of instruction set name ('sse, 'avx', 'avx512') and data type ('float', 'double')

    GPU specific Parameters
    :param gpu_indexing: either 'block' or 'line' , or custom indexing class (see gpucuda/indexing.py)
    :param gpu_indexing_params: dict with indexing parameters (constructor parameters of indexing class)
                              e.g. for 'block' one can specify {'blockSize': (20, 20, 10) }

    :return: abstract syntax tree object, that can either be printed as source code or can be compiled with
             through its compile() function
    """

    # ----  Normalizing parameters
    split_groups = ()
    if isinstance(equations, AssignmentCollection):
        if 'split_groups' in equations.simplification_hints:
            split_groups = equations.simplification_hints['split_groups']
        equations = equations.all_assignments

    # ----  Creating ast
    if target == 'cpu':
        from pystencils.cpu import create_kernel
        from pystencils.cpu import add_openmp
        ast = create_kernel(equations, type_info=data_type, split_groups=split_groups,
                            iteration_slice=iteration_slice, ghost_layers=ghost_layers)
        if cpu_openmp:
            add_openmp(ast, num_threads=cpu_openmp)
        if cpu_vectorize_info:
            import pystencils.backends.simd_instruction_sets as vec
            from pystencils.vectorization import vectorize
            vec_params = cpu_vectorize_info
            vec.selectedInstructionSet = vec.x86_vector_instruction_set(instruction_set=vec_params[0],
                                                                        data_type=vec_params[1])
            vectorize(ast)
        return ast
    elif target == 'llvm':
        from pystencils.llvm import create_kernel
        ast = create_kernel(equations, type_info=data_type, split_groups=split_groups,
                            iteration_slice=iteration_slice, ghost_layers=ghost_layers)
        return ast
    elif target == 'gpu':
        from pystencils.gpucuda import create_cuda_kernel
        ast = create_cuda_kernel(equations, type_info=data_type,
                                 indexing_creator=indexing_creator_from_params(gpu_indexing, gpu_indexing_params),
                                 iteration_slice=iteration_slice, ghost_layers=ghost_layers)
        return ast
    else:
        raise ValueError("Unknown target %s. Has to be one of 'cpu', 'gpu' or 'llvm' " % (target,))


def create_indexed_kernel(assignments, index_fields, target='cpu', data_type="double", coordinate_names=('x', 'y', 'z'),
                          cpu_openmp=True, gpu_indexing='block', gpu_indexing_params={}):
    """
    Similar to :func:`create_kernel`, but here not all cells of a field are updated but only cells with
    coordinates which are stored in an index field. This traversal method can e.g. be used for boundary handling.

    The coordinates are stored in a separated indexField, which is a one dimensional array with struct data type.
    This struct has to contain fields named 'x', 'y' and for 3D fields ('z'). These names are configurable with the
    'coordinate_names' parameter. The struct can have also other fields that can be read and written in the kernel, for
    example boundary parameters.

    index_fields: list of index fields, i.e. 1D fields with struct data type
    coordinate_names: name of the coordinate fields in the struct data type
    """

    if isinstance(assignments, AssignmentCollection):
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
        ast = created_indexed_cuda_kernel(assignments, index_fields, type_info=data_type, coordinate_names=coordinate_names,
                                          indexing_creator=indexing_creator_from_params(gpu_indexing, gpu_indexing_params))
        return ast
    else:
        raise ValueError("Unknown target %s. Has to be either 'cpu' or 'gpu'" % (target,))
