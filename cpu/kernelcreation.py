import sympy as sp
from functools import partial
from pystencils.astnodes import SympyAssignment, Block, LoopOverCoordinate, KernelFunction
from pystencils.transformations import resolveBufferAccesses, resolveFieldAccesses, makeLoopOverDomain, \
    typeAllEquations, getOptimalLoopOrdering, parseBasePointerInfo, moveConstantsBeforeLoop, splitInnerLoop, \
    substituteArrayAccessesWithConstants
from pystencils.data_types import TypedSymbol, BasicType, StructType, create_type
from pystencils.field import Field, FieldType
import pystencils.astnodes as ast
from pystencils.cpu.cpujit import makePythonFunction


def createKernel(listOfEquations, functionName="kernel", typeForSymbol='double', splitGroups=(),
                 iterationSlice=None, ghostLayers=None):
    """
    Creates an abstract syntax tree for a kernel function, by taking a list of update rules.

    Loops are created according to the field accesses in the equations.

    :param listOfEquations: list of sympy equations, containing accesses to :class:`pystencils.field.Field`.
           Defining the update rules of the kernel
    :param functionName: name of the generated function - only important if generated code is written out
    :param typeForSymbol: a map from symbol name to a C type specifier. If not specified all symbols are assumed to
           be of type 'double' except symbols which occur on the left hand side of equations where the
           right hand side is a sympy Boolean which are assumed to be 'bool' .
    :param splitGroups: Specification on how to split up inner loop into multiple loops. For details see
           transformation :func:`pystencils.transformation.splitInnerLoop`
    :param iterationSlice: if not None, iteration is done only over this slice of the field
    :param ghostLayers: a sequence of pairs for each coordinate with lower and upper nr of ghost layers
                        if None, the number of ghost layers is determined automatically and assumed to be equal for a
                        all dimensions

    :return: :class:`pystencils.ast.KernelFunction` node
    """

    def type_symbol(term):
        if isinstance(term, Field.Access) or isinstance(term, TypedSymbol):
            return term
        elif isinstance(term, sp.Symbol):
            if not hasattr(typeForSymbol, '__getitem__'):
                return TypedSymbol(term.name, create_type(typeForSymbol))
            else:
                return TypedSymbol(term.name, typeForSymbol[term.name])
        else:
            raise ValueError("Term has to be field access or symbol")

    fields_read, fields_written, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    all_fields = fields_read.union(fields_written)
    read_only_fields = set([f.name for f in fields_read - fields_written])

    buffers = set([f for f in all_fields if FieldType.isBuffer(f)])
    fields_without_buffers = all_fields - buffers

    body = ast.Block(assignments)
    loop_order = getOptimalLoopOrdering(fields_without_buffers)
    code, loop_strides, loop_vars = makeLoopOverDomain(body, functionName, iterationSlice=iterationSlice,
                                                       ghostLayers=ghostLayers, loopOrder=loop_order)
    code.target = 'cpu'

    if splitGroups:
        typed_split_groups = [[type_symbol(s) for s in splitGroup] for splitGroup in splitGroups]
        splitInnerLoop(code, typed_split_groups)

    base_pointer_info = [['spatialInner0'], ['spatialInner1']] if len(loop_order) >= 2 else [['spatialInner0']]
    base_pointer_infos = {field.name: parseBasePointerInfo(base_pointer_info, loop_order, field)
                          for field in fields_without_buffers}

    buffer_base_pointer_infos = {field.name: parseBasePointerInfo([['spatialInner0']], [0], field) for field in buffers}
    base_pointer_infos.update(buffer_base_pointer_infos)

    base_buffer_index = loop_vars[0]
    stride = 1
    for idx, var in enumerate(loop_vars[1:]):
        cur_stride = loop_strides[idx]
        stride *= int(cur_stride) if isinstance(cur_stride, float) else cur_stride
        base_buffer_index += var * stride

    resolveBufferAccesses(code, base_buffer_index, read_only_fields)
    resolveFieldAccesses(code, read_only_fields, field_to_base_pointer_info=base_pointer_infos)
    substituteArrayAccessesWithConstants(code)
    moveConstantsBeforeLoop(code)
    code.compile = partial(makePythonFunction, code)
    return code


def createIndexedKernel(listOfEquations, indexFields, functionName="kernel", typeForSymbol=None,
                        coordinateNames=('x', 'y', 'z')):
    """
    Similar to :func:`createKernel`, but here not all cells of a field are updated but only cells with
    coordinates which are stored in an index field. This traversal method can e.g. be used for boundary handling.

    The coordinates are stored in a separated indexField, which is a one dimensional array with struct data type.
    This struct has to contain fields named 'x', 'y' and for 3D fields ('z'). These names are configurable with the
    'coordinateNames' parameter. The struct can have also other fields that can be read and written in the kernel, for
    example boundary parameters.

    :param listOfEquations: list of update equations or AST nodes
    :param indexFields: list of index fields, i.e. 1D fields with struct data type
    :param typeForSymbol: see documentation of :func:`createKernel`
    :param functionName: see documentation of :func:`createKernel`
    :param coordinateNames: name of the coordinate fields in the struct data type
    :return: abstract syntax tree
    """
    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    allFields = fieldsRead.union(fieldsWritten)

    for indexField in indexFields:
        indexField.fieldType = FieldType.INDEXED
        assert FieldType.isIndexed(indexField)
        assert indexField.spatialDimensions == 1, "Index fields have to be 1D"

    nonIndexFields = [f for f in allFields if f not in indexFields]
    spatialCoordinates = {f.spatialDimensions for f in nonIndexFields}
    assert len(spatialCoordinates) == 1, "Non-index fields do not have the same number of spatial coordinates"
    spatialCoordinates = list(spatialCoordinates)[0]

    def getCoordinateSymbolAssignment(name):
        for indexField in indexFields:
            assert isinstance(indexField.dtype, StructType), "Index fields have to have a struct datatype"
            dataType = indexField.dtype
            if dataType.has_element(name):
                rhs = indexField[0](name)
                lhs = TypedSymbol(name, BasicType(dataType.get_element_type(name)))
                return SympyAssignment(lhs, rhs)
        raise ValueError("Index %s not found in any of the passed index fields" % (name,))

    coordinateSymbolAssignments = [getCoordinateSymbolAssignment(n) for n in coordinateNames[:spatialCoordinates]]
    coordinateTypedSymbols = [eq.lhs for eq in coordinateSymbolAssignments]
    assignments = coordinateSymbolAssignments + assignments

    # make 1D loop over index fields
    loopBody = Block([])
    loopNode = LoopOverCoordinate(loopBody, coordinate_to_loop_over=0, start=0, stop=indexFields[0].shape[0])

    for assignment in assignments:
        loopBody.append(assignment)

    functionBody = Block([loopNode])
    ast = KernelFunction(functionBody, backend="cpu", function_name=functionName)

    fixedCoordinateMapping = {f.name: coordinateTypedSymbols for f in nonIndexFields}
    resolveFieldAccesses(ast, set(['indexField']), field_to_fixed_coordinates=fixedCoordinateMapping)
    substituteArrayAccessesWithConstants(ast)
    moveConstantsBeforeLoop(ast)
    ast.compile = partial(makePythonFunction, ast)
    return ast


def addOpenMP(astNode, schedule="static", numThreads=True):
    """
    Parallelizes the outer loop with OpenMP

    :param astNode: abstract syntax tree created e.g. by :func:`createKernel`
    :param schedule: OpenMP scheduling policy e.g. 'static' or 'dynamic'
    :param numThreads: explicitly specify number of threads
    """
    if not numThreads:
        return

    assert type(astNode) is ast.KernelFunction
    body = astNode.body
    threadsClause = "" if numThreads and isinstance(numThreads,bool) else " num_threads(%s)" % (numThreads,)
    wrapperBlock = ast.PragmaBlock('#pragma omp parallel' + threadsClause, body.take_child_nodes())
    body.append(wrapperBlock)

    outerLoops = [l for l in body.atoms(ast.LoopOverCoordinate) if l.is_outermost_loop]
    assert outerLoops, "No outer loop found"
    assert len(outerLoops) <= 1, "More than one outer loop found. Which one should be parallelized?"
    loopToParallelize = outerLoops[0]
    try:
        loopRange = int(loopToParallelize.stop - loopToParallelize.start)
    except TypeError:
        loopRange = None

    if numThreads is None:
        import multiprocessing
        numThreads = multiprocessing.cpu_count()

    if loopRange is not None and loopRange < numThreads:
        containedLoops = [l for l in loopToParallelize.body.args if isinstance(l, LoopOverCoordinate)]
        if len(containedLoops) == 1:
            containedLoop = containedLoops[0]
            try:
                containedLoopRange = int(containedLoop.stop - containedLoop.start)
                if containedLoopRange > loopRange:
                    loopToParallelize = containedLoop
            except TypeError:
                pass

    loopToParallelize.prefixLines.append("#pragma omp for schedule(%s)" % (schedule,))
