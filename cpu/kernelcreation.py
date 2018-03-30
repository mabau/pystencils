import sympy as sp
from functools import partial
from pystencils.astnodes import SympyAssignment, Block, LoopOverCoordinate, KernelFunction
from pystencils.transformations import resolveBufferAccesses, resolveFieldAccesses, makeLoopOverDomain, \
    typeAllEquations, getOptimalLoopOrdering, parseBasePointerInfo, moveConstantsBeforeLoop, splitInnerLoop, \
    substituteArrayAccessesWithConstants
from pystencils.data_types import TypedSymbol, BasicType, StructType, createType
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
                return TypedSymbol(term.name, createType(typeForSymbol))
            else:
                return TypedSymbol(term.name, typeForSymbol[term.name])
        else:
            raise ValueError("Term has to be field access or symbol")

    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    allFields = fieldsRead.union(fieldsWritten)
    readOnlyFields = set([f.name for f in fieldsRead - fieldsWritten])

    buffers = set([f for f in allFields if FieldType.isBuffer(f)])
    fieldsWithoutBuffers = allFields - buffers

    body = ast.Block(assignments)
    loopOrder = getOptimalLoopOrdering(fieldsWithoutBuffers)
    code, loopStrides, loopVars = makeLoopOverDomain(body, functionName, iterationSlice=iterationSlice,
                                                     ghostLayers=ghostLayers, loopOrder=loopOrder)
    code.target = 'cpu'

    if splitGroups:
        typedSplitGroups = [[type_symbol(s) for s in splitGroup] for splitGroup in splitGroups]
        splitInnerLoop(code, typedSplitGroups)

    basePointerInfo = [['spatialInner0'], ['spatialInner1']] if len(loopOrder) >= 2 else [['spatialInner0']]
    basePointerInfos = {field.name: parseBasePointerInfo(basePointerInfo, loopOrder, field)
                        for field in fieldsWithoutBuffers}

    bufferBasePointerInfos = {field.name: parseBasePointerInfo([['spatialInner0']], [0], field) for field in buffers}
    basePointerInfos.update(bufferBasePointerInfos)

    baseBufferIndex = loopVars[0]
    stride = 1
    for idx, var in enumerate(loopVars[1:]):
        curStride = loopStrides[idx]
        stride *= int(curStride) if isinstance(curStride, float) else curStride
        baseBufferIndex += var * stride

    resolveBufferAccesses(code, baseBufferIndex, readOnlyFields)
    resolveFieldAccesses(code, readOnlyFields, fieldToBasePointerInfo=basePointerInfos)
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
            if dataType.hasElement(name):
                rhs = indexField[0](name)
                lhs = TypedSymbol(name, BasicType(dataType.getElementType(name)))
                return SympyAssignment(lhs, rhs)
        raise ValueError("Index %s not found in any of the passed index fields" % (name,))

    coordinateSymbolAssignments = [getCoordinateSymbolAssignment(n) for n in coordinateNames[:spatialCoordinates]]
    coordinateTypedSymbols = [eq.lhs for eq in coordinateSymbolAssignments]
    assignments = coordinateSymbolAssignments + assignments

    # make 1D loop over index fields
    loopBody = Block([])
    loopNode = LoopOverCoordinate(loopBody, coordinateToLoopOver=0, start=0, stop=indexFields[0].shape[0])

    for assignment in assignments:
        loopBody.append(assignment)

    functionBody = Block([loopNode])
    ast = KernelFunction(functionBody, backend="cpu", functionName=functionName)

    fixedCoordinateMapping = {f.name: coordinateTypedSymbols for f in nonIndexFields}
    resolveFieldAccesses(ast, set(['indexField']), fieldToFixedCoordinates=fixedCoordinateMapping)
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
    wrapperBlock = ast.PragmaBlock('#pragma omp parallel' + threadsClause, body.takeChildNodes())
    body.append(wrapperBlock)

    outerLoops = [l for l in body.atoms(ast.LoopOverCoordinate) if l.isOutermostLoop]
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
