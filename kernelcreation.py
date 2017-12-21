from pystencils.equationcollection import EquationCollection
from pystencils.gpucuda.indexing import indexingCreatorFromParams


def createKernel(equations, target='cpu', dataType="double", iterationSlice=None, ghostLayers=None,
                 cpuOpenMP=True, cpuVectorizeInfo=None,
                 gpuIndexing='block', gpuIndexingParams={}):
    """
    Creates abstract syntax tree (AST) of kernel, using a list of update equations.
    :param equations: either be a plain list of equations or a EquationCollection object
    :param target: 'cpu', 'llvm' or 'gpu'
    :param dataType: data type used for all untyped symbols (i.e. non-fields), can also be a dict from symbol name
                     to type
    :param iterationSlice: rectangular subset to iterate over, if not specified the complete non-ghost layer part of the
                           field is iterated over
    :param ghostLayers: if left to default, the number of necessary ghost layers is determined automatically
                        a single integer specifies the ghost layer count at all borders, can also be a sequence of
                        pairs [(xLowerGl, xUpperGl), .... ]

    CPU specific Parameters:
    :param cpuOpenMP: True or number of threads for OpenMP parallelization, False for no OpenMP
    :param cpuVectorizeInfo: pair of instruction set name ('sse, 'avx', 'avx512') and data type ('float', 'double')

    GPU specific Parameters
    :param gpuIndexing: either 'block' or 'line' , or custom indexing class (see gpucuda/indexing.py)
    :param gpuIndexingParams: dict with indexing parameters (constructor parameters of indexing class)
                              e.g. for 'block' one can specify {'blockSize': (20, 20, 10) }

    :return: abstract syntax tree object, that can either be printed as source code or can be compiled with
             through its compile() function
    """

    # ----  Normalizing parameters
    splitGroups = ()
    if isinstance(equations, EquationCollection):
        if 'splitGroups' in equations.simplificationHints:
            splitGroups = equations.simplificationHints['splitGroups']
        equations = equations.allEquations

    # ----  Creating ast
    if target == 'cpu':
        from pystencils.cpu import createKernel
        from pystencils.cpu import addOpenMP
        ast = createKernel(equations, typeForSymbol=dataType, splitGroups=splitGroups,
                           iterationSlice=iterationSlice, ghostLayers=ghostLayers)
        if cpuOpenMP:
            addOpenMP(ast, numThreads=cpuOpenMP)
        if cpuVectorizeInfo:
            import pystencils.backends.simd_instruction_sets as vec
            from pystencils.vectorization import vectorize
            vecParams = cpuVectorizeInfo
            vec.selectedInstructionSet = vec.x86VectorInstructionSet(instructionSet=vecParams[0], dataType=vecParams[1])
            vectorize(ast)
        return ast
    elif target == 'llvm':
        from pystencils.llvm import createKernel
        ast = createKernel(equations, typeForSymbol=dataType, splitGroups=splitGroups,
                           iterationSlice=iterationSlice, ghostLayers=ghostLayers)
        return ast
    elif target == 'gpu':
        from pystencils.gpucuda import createCUDAKernel
        ast = createCUDAKernel(equations, typeForSymbol=dataType,
                               indexingCreator=indexingCreatorFromParams(gpuIndexing, gpuIndexingParams),
                               iterationSlice=iterationSlice, ghostLayers=ghostLayers)
        return ast
    else:
        raise ValueError("Unknown target %s. Has to be one of 'cpu', 'gpu' or 'llvm' " % (target,))


def createIndexedKernel(equations, indexFields, target='cpu', dataType="double", coordinateNames=('x', 'y', 'z'),
                        cpuOpenMP=True,
                        gpuIndexing='block', gpuIndexingParams={}):
    """
    Similar to :func:`createKernel`, but here not all cells of a field are updated but only cells with
    coordinates which are stored in an index field. This traversal method can e.g. be used for boundary handling.

    The coordinates are stored in a separated indexField, which is a one dimensional array with struct data type.
    This struct has to contain fields named 'x', 'y' and for 3D fields ('z'). These names are configurable with the
    'coordinateNames' parameter. The struct can have also other fields that can be read and written in the kernel, for
    example boundary parameters.

    indexFields: list of index fields, i.e. 1D fields with struct data type
    coordinateNames: name of the coordinate fields in the struct data type
    """

    if isinstance(equations, EquationCollection):
        equations = equations.allEquations
    if target == 'cpu':
        from pystencils.cpu import createIndexedKernel
        from pystencils.cpu import addOpenMP
        ast = createIndexedKernel(equations, indexField=indexFields, typeForSymbol=dataType,
                                  coordinateNames=coordinateNames)
        if cpuOpenMP:
            addOpenMP(ast, numThreads=cpuOpenMP)
        return ast
    elif target == 'llvm':
        raise NotImplementedError("Indexed kernels are not yet supported in LLVM backend")
    elif target == 'gpu':
        from pystencils.gpucuda import createdIndexedCUDAKernel
        ast = createdIndexedCUDAKernel(equations, indexFields, typeForSymbol=dataType, coordinateNames=coordinateNames,
                                       indexingCreator=indexingCreatorFromParams(gpuIndexing, gpuIndexingParams))
        return ast
    else:
        raise ValueError("Unknown target %s. Has to be either 'cpu' or 'gpu'" % (target,))
