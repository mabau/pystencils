from collections import defaultdict
import sympy as sp
from sympy.logic.boolalg import Boolean
from sympy.tensor import IndexedBase
from pystencils.field import Field, offsetComponentToDirectionString
from pystencils.typedsymbol import TypedSymbol
import pystencils.ast as ast


def makeLoopOverDomain(body, functionName):
    """
    Uses :class:`pystencils.field.Field.Access` to create (multiple) loops around given AST.
    :param body: list of nodes
    :param functionName: name of generated C function
    :return: :class:`LoopOverCoordinate` instance with nested loops, ordered according to field layouts
    """
    # find correct ordering by inspecting participating FieldAccesses
    fieldAccesses = body.atoms(Field.Access)
    fieldList = [e.field for e in fieldAccesses]
    fields = set(fieldList)
    loopOrder = getOptimalLoopOrdering(fields)

    # find number of required ghost layers
    requiredGhostLayers = max([fa.requiredGhostLayers for fa in fieldAccesses])

    shapes = set([f.spatialShape for f in fields])

    if len(shapes) > 1:
        nrOfFixedSizedFields = 0
        for shape in shapes:
            if not isinstance(shape[0], sp.Basic):
                nrOfFixedSizedFields += 1
        assert nrOfFixedSizedFields <= 1, "Differently sized field accesses in loop body: " + str(shapes)
    shape = list(shapes)[0]

    currentBody = body
    lastLoop = None
    for i, loopCoordinate in enumerate(loopOrder):
        newLoop = ast.LoopOverCoordinate(currentBody, loopCoordinate, shape, 1, requiredGhostLayers,
                                         isInnermostLoop=(i == 0), isOutermostLoop=(i == len(loopOrder) - 1))
        lastLoop = newLoop
        currentBody = ast.Block([lastLoop])
    return ast.KernelFunction(currentBody, functionName)


def createIntermediateBasePointer(fieldAccess, coordinates, previousPtr):
    r"""
    Addressing elements in structured arrays are done with :math:`ptr\left[ \sum_i c_i \cdot s_i \right]`
    where :math:`c_i` is the coordinate value and :math:`s_i` the stride of a coordinate.
    The sum can be split up into multiple parts, such that parts of it can be pulled before loops.
    This function creates such an access for coordinates :math:`i \in \mbox{coordinates}`.
    Returns a new typed symbol, where the name encodes which coordinates have been resolved.
    :param fieldAccess: instance of :class:`pystencils.field.Field.Access` which provides strides and offsets
    :param coordinates: mapping of coordinate ids to its value, where stride*value is calculated
    :param previousPtr: the pointer which is dereferenced
    :return: tuple with the new pointer symbol and the calculated offset

    Example:
        >>> field = Field.createGeneric('myfield', spatialDimensions=2, indexDimensions=1)
        >>> x, y = sp.symbols("x y")
        >>> prevPointer = TypedSymbol("ptr", "double")
        >>> createIntermediateBasePointer(field[1,-2](5), {0: x}, prevPointer)
        (ptr_E, x*fstride_myfield[0] + fstride_myfield[0])
        >>> createIntermediateBasePointer(field[1,-2](5), {0: x, 1 : y }, prevPointer)
        (ptr_E_2S, x*fstride_myfield[0] + y*fstride_myfield[1] + fstride_myfield[0] - 2*fstride_myfield[1])
    """
    field = fieldAccess.field

    offset = 0
    name = ""
    listToHash = []
    for coordinateId, coordinateValue in coordinates.items():
        offset += field.strides[coordinateId] * coordinateValue

        if coordinateId < field.spatialDimensions:
            offset += field.strides[coordinateId] * fieldAccess.offsets[coordinateId]
            if type(fieldAccess.offsets[coordinateId]) is int:
                offsetComp = offsetComponentToDirectionString(coordinateId, fieldAccess.offsets[coordinateId])
                name += "_"
                name += offsetComp if offsetComp else "C"
            else:
                listToHash.append(fieldAccess.offsets[coordinateId])
        else:
            if type(coordinateValue) is int:
                name += "_%d" % (coordinateValue,)
            else:
                listToHash.append(coordinateValue)

    if len(listToHash) > 0:
        name += "%0.6X" % (abs(hash(tuple(listToHash))))

    newPtr = TypedSymbol(previousPtr.name + name, previousPtr.dtype)
    return newPtr, offset


def parseBasePointerInfo(basePointerSpecification, loopOrder, field):
    """
    Creates base pointer specification for :func:`resolveFieldAccesses` function.

    Specification of how many and which intermediate pointers are created for a field access.
    For example [ (0), (2,3,)]  creates on base pointer for coordinates 2 and 3 and writes the offset for coordinate
    zero directly in the field access. These specifications are more sensible defined dependent on the loop ordering.
    This function translates more readable version into the specification above.

    Allowed specifications:
        - "spatialInner<int>" spatialInner0 is the innermost loop coordinate,
          spatialInner1 the loop enclosing the innermost
        - "spatialOuter<int>" spatialOuter0 is the outermost loop
        - "index<int>": index coordinate
        - "<int>": specifying directly the coordinate

    :param basePointerSpecification: nested list with above specifications
    :param loopOrder: list with ordering of loops from inner to outer
    :param field:
    :return: list of tuples that can be passed to :func:`resolveFieldAccesses`
    """
    result = []
    specifiedCoordinates = set()
    for specGroup in basePointerSpecification:
        newGroup = []

        def addNewElement(i):
            if i >= field.spatialDimensions + field.indexDimensions:
                raise ValueError("Coordinate %d does not exist" % (i,))
            newGroup.append(i)
            if i in specifiedCoordinates:
                raise ValueError("Coordinate %d specified two times" % (i,))
            specifiedCoordinates.add(i)

        for element in specGroup:
            if type(element) is int:
                addNewElement(element)
            elif element.startswith("spatial"):
                element = element[len("spatial"):]
                if element.startswith("Inner"):
                    index = int(element[len("Inner"):])
                    addNewElement(loopOrder[index])
                elif element.startswith("Outer"):
                    index = int(element[len("Outer"):])
                    addNewElement(loopOrder[-index])
                elif element == "all":
                    for i in range(field.spatialDimensions):
                        addNewElement(i)
                else:
                    raise ValueError("Could not parse " + element)
            elif element.startswith("index"):
                index = int(element[len("index"):])
                addNewElement(field.spatialDimensions + index)
            else:
                raise ValueError("Unknown specification %s" % (element,))

        result.append(newGroup)

    allCoordinates = set(range(field.spatialDimensions + field.indexDimensions))
    rest = allCoordinates - specifiedCoordinates
    if rest:
        result.append(list(rest))
    return result


def resolveFieldAccesses(astNode, fieldToBasePointerInfo={}, fieldToFixedCoordinates={}):
    """
    Substitutes :class:`pystencils.field.Field.Access` nodes by array indexing

    :param astNode: the AST root
    :param fieldToBasePointerInfo: a list of tuples indicating which intermediate base pointers should be created
                                   for details see :func:`parseBasePointerInfo`
    :param fieldToFixedCoordinates: map of field name to a tuple of coordinate symbols. Instead of using the loop
                                    counters to index the field these symbols are used as coordinates
    :return: transformed AST
    """
    def visitSympyExpr(expr, enclosingBlock):
        if isinstance(expr, Field.Access):
            fieldAccess = expr
            field = fieldAccess.field
            if field.name in fieldToBasePointerInfo:
                basePointerInfo = fieldToBasePointerInfo[field.name]
            else:
                basePointerInfo = [list(range(field.indexDimensions + field.spatialDimensions))]

            dtype = "%s * __restrict__" % field.dtype
            if field.readOnly:
                dtype = "const " + dtype

            fieldPtr = TypedSymbol("%s%s" % (Field.DATA_PREFIX, field.name), dtype)

            lastPointer = fieldPtr

            def createCoordinateDict(group):
                coordDict = {}
                for e in group:
                    if e < field.spatialDimensions:
                        if field.name in fieldToFixedCoordinates:
                            coordDict[e] = fieldToFixedCoordinates[field.name][e]
                        else:
                            ctrName = ast.LoopOverCoordinate.LOOP_COUNTER_NAME_PREFIX
                            coordDict[e] = TypedSymbol("%s_%d" % (ctrName, e), "int")
                    else:
                        coordDict[e] = fieldAccess.index[e-field.spatialDimensions]
                return coordDict

            for group in reversed(basePointerInfo[1:]):
                coordDict = createCoordinateDict(group)
                newPtr, offset = createIntermediateBasePointer(fieldAccess, coordDict, lastPointer)
                if newPtr not in enclosingBlock.symbolsDefined:
                    enclosingBlock.insertFront(ast.SympyAssignment(newPtr, lastPointer + offset, isConst=False))
                lastPointer = newPtr

            _, offset = createIntermediateBasePointer(fieldAccess, createCoordinateDict(basePointerInfo[0]),
                                                      lastPointer)
            baseArr = IndexedBase(lastPointer, shape=(1,))
            return baseArr[offset]
        else:
            newArgs = [visitSympyExpr(e, enclosingBlock) for e in expr.args]
            kwargs = {'evaluate': False} if type(expr) is sp.Add or type(expr) is sp.Mul else {}
            return expr.func(*newArgs, **kwargs) if newArgs else expr

    def visitNode(subAst):
        if isinstance(subAst, ast.SympyAssignment):
            enclosingBlock = subAst.parent
            assert type(enclosingBlock) is ast.Block
            subAst.lhs = visitSympyExpr(subAst.lhs, enclosingBlock)
            subAst.rhs = visitSympyExpr(subAst.rhs, enclosingBlock)
        else:
            for i, a in enumerate(subAst.args):
                visitNode(a)

    return visitNode(astNode)


def moveConstantsBeforeLoop(astNode):
    """
    Moves :class:`pystencils.ast.SympyAssignment` nodes out of loop body if they are iteration independent.
    Call this after creating the loop structure with :func:`makeLoopOverDomain`
    :param astNode:
    :return:
    """
    def findBlockToMoveTo(node):
        """Traverses parents of node as long as the symbols are independent and returns a (parent) block
        the assignment can be safely moved to
        :param node: SympyAssignment inside a Block"""
        assert isinstance(node, ast.SympyAssignment)
        assert isinstance(node.parent, ast.Block)

        lastBlock = node.parent
        element = node.parent
        while element:
            if isinstance(element, ast.Block):
                lastBlock = element
            if node.symbolsRead.intersection(element.symbolsDefined):
                break
            element = element.parent
        return lastBlock

    def checkIfAssignmentAlreadyInBlock(assignment, targetBlock):
        for arg in targetBlock.args:
            if type(arg) is not ast.SympyAssignment:
                continue
            if arg.lhs == assignment.lhs:
                return arg
        return None

    for block in astNode.atoms(ast.Block):
        children = block.takeChildNodes()
        for child in children:
            if not isinstance(child, ast.SympyAssignment):
                block.append(child)
            else:
                target = findBlockToMoveTo(child)
                if target == block:     # movement not possible
                    target.append(child)
                else:
                    existingAssignment = checkIfAssignmentAlreadyInBlock(child, target)
                    if not existingAssignment:
                        target.insertFront(child)
                    else:
                        assert existingAssignment.rhs == child.rhs, "Symbol with same name exists already"


def splitInnerLoop(astNode, symbolGroups):
    """
    Splits inner loop into multiple loops to minimize the amount of simultaneous load/store streams
    :param astNode: AST root
    :param symbolGroups: sequence of symbol sequences: for each symbol sequence a new inner loop is created which
    updates these symbols and their dependent symbols. Symbols which are in none of the symbolGroups and which
    no symbol in a symbol group depends on, are not updated!
    :return: transformed AST
    """
    allLoops = astNode.atoms(ast.LoopOverCoordinate)
    innerLoop = [l for l in allLoops if l.isInnermostLoop]
    assert len(innerLoop) == 1, "Error in AST: multiple innermost loops. Was split transformation already called?"
    innerLoop = innerLoop[0]
    assert type(innerLoop.body) is ast.Block
    outerLoop = [l for l in allLoops if l.isOutermostLoop]
    assert len(outerLoop) == 1, "Error in AST, multiple outermost loops."
    outerLoop = outerLoop[0]

    symbolsWithTemporaryArray = dict()

    assignmentMap = {a.lhs: a for a in innerLoop.body.args}

    assignmentGroups = []
    for symbolGroup in symbolGroups:
        # get all dependent symbols
        symbolsToProcess = list(symbolGroup)
        symbolsResolved = set()
        while symbolsToProcess:
            s = symbolsToProcess.pop()
            if s in symbolsResolved:
                continue

            if s in assignmentMap:  # if there is no assignment inside the loop body it is independent already
                for newSymbol in assignmentMap[s].rhs.atoms(sp.Symbol):
                    if type(newSymbol) is not Field.Access and newSymbol not in symbolsWithTemporaryArray:
                        symbolsToProcess.append(newSymbol)
            symbolsResolved.add(s)

        for symbol in symbolGroup:
            if type(symbol) is not Field.Access:
                assert type(symbol) is TypedSymbol
                symbolsWithTemporaryArray[symbol] = IndexedBase(symbol, shape=(1,))[innerLoop.loopCounterSymbol]

        assignmentGroup = []
        for assignment in innerLoop.body.args:
            if assignment.lhs in symbolsResolved:
                newRhs = assignment.rhs.subs(symbolsWithTemporaryArray.items())
                if type(assignment.lhs) is not Field.Access and assignment.lhs in symbolGroup:
                    newLhs = IndexedBase(assignment.lhs, shape=(1,))[innerLoop.loopCounterSymbol]
                else:
                    newLhs = assignment.lhs
                assignmentGroup.append(ast.SympyAssignment(newLhs, newRhs))
        assignmentGroups.append(assignmentGroup)

    newLoops = [innerLoop.newLoopWithDifferentBody(ast.Block(group)) for group in assignmentGroups]
    innerLoop.parent.replace(innerLoop, newLoops)

    for tmpArray in symbolsWithTemporaryArray:
        outerLoop.parent.insertFront(ast.TemporaryMemoryAllocation(tmpArray, innerLoop.iterationRegionWithGhostLayer))
        outerLoop.parent.append(ast.TemporaryMemoryFree(tmpArray))


def typeAllEquations(eqs, typeForSymbol):
    """
    Traverses AST and replaces every :class:`sympy.Symbol` by a :class:`pystencils.typedsymbol.TypedSymbol`.
    Additionally returns sets of all fields which are read/written

    :param eqs: list of equations
    :param typeForSymbol: dict mapping symbol names to types. Types are strings of C types like 'int' or 'double'
    :return: ``fieldsRead, fieldsWritten, typedEquations`` set of read fields, set of written fields, list of equations
               where symbols have been replaced by typed symbols
    """
    fieldsWritten = set()
    fieldsRead = set()

    def processRhs(term):
        """Replaces Symbols by:
            - TypedSymbol if symbol is not a field access
        """
        if isinstance(term, Field.Access):
            fieldsRead.add(term.field)
            return term
        elif isinstance(term, sp.Symbol):
            return TypedSymbol(term.name, typeForSymbol[term.name])
        else:
            newArgs = [processRhs(arg) for arg in term.args]
            return term.func(*newArgs) if newArgs else term

    def processLhs(term):
        """Replaces symbol by TypedSymbol and adds field to fieldsWriten"""
        if isinstance(term, Field.Access):
            fieldsWritten.add(term.field)
            return term
        elif isinstance(term, sp.Symbol):
            return TypedSymbol(term.name, typeForSymbol[term.name])
        else:
            assert False, "Expected a symbol as left-hand-side"

    typedEquations = []
    for eq in eqs:
        if isinstance(eq, sp.Eq):
            newLhs = processLhs(eq.lhs)
            newRhs = processRhs(eq.rhs)
            typedEquations.append(ast.SympyAssignment(newLhs, newRhs))
        else:
            assert isinstance(eq, ast.Node), "Only equations and ast nodes are allowed in input"
            typedEquations.append(eq)

    typedEquations = typedEquations

    return fieldsRead, fieldsWritten, typedEquations


# --------------------------------------- Helper Functions -------------------------------------------------------------


def typingFromSympyInspection(eqs, defaultType="double"):
    """
    Creates a default symbol name to type mapping.
    If a sympy Boolean is assigned to a symbol it is assumed to be 'bool' otherwise the default type, usually ('double')
    :param eqs: list of equations
    :param defaultType: the type for non-boolean symbols
    :return: dictionary, mapping symbol name to type
    """
    result = defaultdict(lambda: defaultType)
    for eq in eqs:
        if isinstance(eq.rhs, Boolean):
            result[eq.lhs.name] = "bool"
    return result


def getNextParentOfType(node, parentType):
    """
    Traverses the AST nodes parents until a parent of given type was found. If no such parent is found, None is returned
    """
    parent = node.parent
    while parent is not None:
        if isinstance(parent, parentType):
            return parent
        parent = parent.parent
    return None


def getOptimalLoopOrdering(fields):
    """
    Determines the optimal loop order for a given set of fields.
    If the fields have different memory layout or different sizes an exception is thrown.
    :param fields: sequence of fields
    :return: list of coordinate ids, where the first list entry should be the innermost loop
    """
    assert len(fields) > 0
    refField = next(iter(fields))
    for field in fields:
        if field.spatialDimensions != refField.spatialDimensions:
            raise ValueError("All fields have to have the same number of spatial dimensions")

    layouts = set([field.layout for field in fields])
    if len(layouts) > 1:
        raise ValueError("Due to different layout of the fields no optimal loop ordering exists")
    layout = list(layouts)[0]
    return list(reversed(layout))


def getLoopHierarchy(astNode):
    """Determines the loop structure around a given AST node.
    :param astNode: the AST node
    :return: list of coordinate ids, where the first list entry is the innermost loop
    """
    result = []
    node = astNode
    while node is not None:
        node = getNextParentOfType(node, ast.LoopOverCoordinate)
        if node:
            result.append(node.coordinateToLoopOver)
    return result