from collections import defaultdict, OrderedDict
from operator import attrgetter
from copy import deepcopy

import sympy as sp
from sympy.logic.boolalg import Boolean
from sympy.tensor import IndexedBase

from pystencils.field import Field, offsetComponentToDirectionString
from pystencils.data_types import TypedSymbol, createType, PointerType, StructType, getBaseType, castFunc
from pystencils.slicing import normalizeSlice
import pystencils.astnodes as ast


def filteredTreeIteration(node, nodeType):
    for arg in node.args:
        if isinstance(arg, nodeType):
            yield arg
        yield from filteredTreeIteration(arg, nodeType)


def fastSubs(term, subsDict):
    """Similar to sympy subs function.
    This version is much faster for big substitution dictionaries than sympy version"""
    def visit(expr):
        if expr in subsDict:
            return subsDict[expr]
        if not hasattr(expr, 'args'):
            return expr
        paramList = [visit(a) for a in expr.args]
        return expr if not paramList else expr.func(*paramList)
    return visit(term)


def getCommonShape(fieldSet):
    """Takes a set of pystencils Fields and returns their common spatial shape if it exists. Otherwise
    ValueError is raised"""
    nrOfFixedShapedFields = 0
    for f in fieldSet:
        if f.hasFixedShape:
            nrOfFixedShapedFields += 1

    if nrOfFixedShapedFields > 0 and nrOfFixedShapedFields != len(fieldSet):
        fixedFieldNames = ",".join([f.name for f in fieldSet if f.hasFixedShape])
        varFieldNames = ",".join([f.name for f in fieldSet if not f.hasFixedShape])
        msg = "Mixing fixed-shaped and variable-shape fields in a single kernel is not possible\n"
        msg += "Variable shaped: %s \nFixed shaped:    %s" % (varFieldNames, fixedFieldNames)
        raise ValueError(msg)

    shapeSet = set([f.spatialShape for f in fieldSet])
    if nrOfFixedShapedFields == len(fieldSet):
        if len(shapeSet) != 1:
            raise ValueError("Differently sized field accesses in loop body: " + str(shapeSet))

    shape = list(sorted(shapeSet, key=lambda e: str(e[0])))[0]
    return shape


def makeLoopOverDomain(body, functionName, iterationSlice=None, ghostLayers=None, loopOrder=None):
    """
    Uses :class:`pystencils.field.Field.Access` to create (multiple) loops around given AST.

    :param body: list of nodes
    :param functionName: name of generated C function
    :param iterationSlice: if not None, iteration is done only over this slice of the field
    :param ghostLayers: a sequence of pairs for each coordinate with lower and upper nr of ghost layers
                if None, the number of ghost layers is determined automatically and assumed to be equal for a
                all dimensions
    :param loopOrder: loop ordering from outer to inner loop (optimal ordering is same as layout)
    :return: :class:`LoopOverCoordinate` instance with nested loops, ordered according to field layouts
    """
    # find correct ordering by inspecting participating FieldAccesses
    fieldAccesses = body.atoms(Field.Access)
    fieldList = [e.field for e in fieldAccesses]
    fields = set(fieldList)

    if loopOrder is None:
        loopOrder = getOptimalLoopOrdering(fields)

    shape = getCommonShape(fields)

    if iterationSlice is not None:
        iterationSlice = normalizeSlice(iterationSlice, shape)

    if ghostLayers is None:
        requiredGhostLayers = max([fa.requiredGhostLayers for fa in fieldAccesses])
        ghostLayers = [(requiredGhostLayers, requiredGhostLayers)] * len(loopOrder)
    if isinstance(ghostLayers, int):
        ghostLayers = [(ghostLayers, ghostLayers)] * len(loopOrder)

    currentBody = body
    lastLoop = None
    for i, loopCoordinate in enumerate(reversed(loopOrder)):
        if iterationSlice is None:
            begin = ghostLayers[loopCoordinate][0]
            end = shape[loopCoordinate] - ghostLayers[loopCoordinate][1]
            newLoop = ast.LoopOverCoordinate(currentBody, loopCoordinate, begin, end, 1)
            lastLoop = newLoop
            currentBody = ast.Block([lastLoop])
        else:
            sliceComponent = iterationSlice[loopCoordinate]
            if type(sliceComponent) is slice:
                sc = sliceComponent
                newLoop = ast.LoopOverCoordinate(currentBody, loopCoordinate, sc.start, sc.stop, sc.step)
                lastLoop = newLoop
                currentBody = ast.Block([lastLoop])
            else:
                assignment = ast.SympyAssignment(ast.LoopOverCoordinate.getLoopCounterSymbol(loopCoordinate),
                                                 sp.sympify(sliceComponent))
                currentBody.insertFront(assignment)
    return ast.KernelFunction(currentBody, ghostLayers=ghostLayers, functionName=functionName)


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
    :param loopOrder: list with ordering of loops from outer to inner
    :param field:
    :return: list of tuples that can be passed to :func:`resolveFieldAccesses`
    """
    result = []
    specifiedCoordinates = set()
    loopOrder = list(reversed(loopOrder))
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


def substituteArrayAccessesWithConstants(astNode):
    """Substitutes all instances of Indexed (array acceses) that are not field accesses with constants.
    Benchmarks showed that using an array access as loop bound or in pointer computations cause some compilers to do 
    less optimizations.  
    This transformation should be after field accesses have been resolved (since they introduce array accesses) and 
    before constants are moved before the loops.
    """

    def handleSympyExpression(expr, parentBlock):
        """Returns sympy expression where array accesses have been replaced with constants, together with a list
        of assignments that define these constants"""
        if not isinstance(expr, sp.Expr):
            return expr

        # get all indexed expressions that are not field accesses
        indexedExprs = [e for e in expr.atoms(sp.Indexed) if not isinstance(e, ast.ResolvedFieldAccess)]

        # special case: right hand side is a single indexed expression, then nothing has to be done
        if len(indexedExprs) == 1 and expr == indexedExprs[0]:
            return expr

        constantsDefinitions = []
        constantSubstitutions = {}
        for indexedExpr in indexedExprs:
            base, idx = indexedExpr.args
            typedSymbol = base.args[0]
            baseType = deepcopy(getBaseType(typedSymbol.dtype))
            baseType.const = False
            constantReplacingIndexed = TypedSymbol(typedSymbol.name + str(idx), baseType)
            constantsDefinitions.append(ast.SympyAssignment(constantReplacingIndexed, indexedExpr))
            constantSubstitutions[indexedExpr] = constantReplacingIndexed
        constantsDefinitions.sort(key=lambda e: e.lhs.name)

        alreadyDefined = parentBlock.symbolsDefined
        for newAssignment in constantsDefinitions:
            if newAssignment.lhs not in alreadyDefined:
                parentBlock.insertBefore(newAssignment, astNode)

        return expr.subs(constantSubstitutions)

    if isinstance(astNode, ast.SympyAssignment):
        astNode.rhs = handleSympyExpression(astNode.rhs, astNode.parent)
        astNode.lhs = handleSympyExpression(astNode.lhs, astNode.parent)
    elif isinstance(astNode, ast.LoopOverCoordinate):
        astNode.start = handleSympyExpression(astNode.start, astNode.parent)
        astNode.stop = handleSympyExpression(astNode.stop, astNode.parent)
        astNode.step = handleSympyExpression(astNode.step, astNode.parent)
        substituteArrayAccessesWithConstants(astNode.body)
    else:
        for a in astNode.args:
            substituteArrayAccessesWithConstants(a)


def resolveFieldAccesses(astNode, readOnlyFieldNames=set(), fieldToBasePointerInfo={}, fieldToFixedCoordinates={}):
    """
    Substitutes :class:`pystencils.field.Field.Access` nodes by array indexing

    :param astNode: the AST root
    :param readOnlyFieldNames: set of field names which are considered read-only
    :param fieldToBasePointerInfo: a list of tuples indicating which intermediate base pointers should be created
                                   for details see :func:`parseBasePointerInfo`
    :param fieldToFixedCoordinates: map of field name to a tuple of coordinate symbols. Instead of using the loop
                                    counters to index the field these symbols are used as coordinates
    :return: transformed AST
    """
    fieldToBasePointerInfo = OrderedDict(sorted(fieldToBasePointerInfo.items(), key=lambda pair: pair[0]))
    fieldToFixedCoordinates = OrderedDict(sorted(fieldToFixedCoordinates.items(), key=lambda pair: pair[0]))

    def visitSympyExpr(expr, enclosingBlock, sympyAssignment):
        if isinstance(expr, Field.Access):
            fieldAccess = expr
            field = fieldAccess.field
            if field.name in fieldToBasePointerInfo:
                basePointerInfo = fieldToBasePointerInfo[field.name]
            else:
                basePointerInfo = [list(range(field.indexDimensions + field.spatialDimensions))]

            dtype = PointerType(field.dtype, const=field.name in readOnlyFieldNames, restrict=True)
            fieldPtr = TypedSymbol("%s%s" % (Field.DATA_PREFIX, symbolNameToVariableName(field.name)), dtype)

            def createCoordinateDict(group):
                coordDict = {}
                for e in group:
                    if e < field.spatialDimensions:
                        if field.name in fieldToFixedCoordinates:
                            coordDict[e] = fieldToFixedCoordinates[field.name][e]
                        else:
                            ctrName = ast.LoopOverCoordinate.LOOP_COUNTER_NAME_PREFIX
                            coordDict[e] = TypedSymbol("%s_%d" % (ctrName, e), 'int')
                        coordDict[e] *= field.dtype.itemSize
                    else:
                        if isinstance(field.dtype, StructType):
                            assert field.indexDimensions == 1
                            accessedFieldName = fieldAccess.index[0]
                            assert isinstance(accessedFieldName, str)
                            coordDict[e] = field.dtype.getElementOffset(accessedFieldName)
                        else:
                            coordDict[e] = fieldAccess.index[e - field.spatialDimensions]
                return coordDict

            lastPointer = fieldPtr

            for group in reversed(basePointerInfo[1:]):
                coordDict = createCoordinateDict(group)
                newPtr, offset = createIntermediateBasePointer(fieldAccess, coordDict, lastPointer)
                if newPtr not in enclosingBlock.symbolsDefined:
                    newAssignment = ast.SympyAssignment(newPtr, lastPointer + offset, isConst=False)
                    enclosingBlock.insertBefore(newAssignment, sympyAssignment)
                lastPointer = newPtr

            coordDict = createCoordinateDict(basePointerInfo[0])
            _, offset = createIntermediateBasePointer(fieldAccess, coordDict, lastPointer)
            result = ast.ResolvedFieldAccess(lastPointer, offset, fieldAccess.field,
                                             fieldAccess.offsets, fieldAccess.index)

            if isinstance(getBaseType(fieldAccess.field.dtype), StructType):
                newType = fieldAccess.field.dtype.getElementType(fieldAccess.index[0])
                result = castFunc(result, newType)
            return visitSympyExpr(result, enclosingBlock, sympyAssignment)
        else:
            if isinstance(expr, ast.ResolvedFieldAccess):
                return expr

            newArgs = [visitSympyExpr(e, enclosingBlock, sympyAssignment) for e in expr.args]
            kwargs = {'evaluate': False} if type(expr) in (sp.Add, sp.Mul, sp.Piecewise) else {}
            return expr.func(*newArgs, **kwargs) if newArgs else expr

    def visitNode(subAst):
        if isinstance(subAst, ast.SympyAssignment):
            enclosingBlock = subAst.parent
            assert type(enclosingBlock) is ast.Block
            subAst.lhs = visitSympyExpr(subAst.lhs, enclosingBlock, subAst)
            subAst.rhs = visitSympyExpr(subAst.rhs, enclosingBlock, subAst)
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
        """
        Traverses parents of node as long as the symbols are independent and returns a (parent) block
        the assignment can be safely moved to
        :param node: SympyAssignment inside a Block
        :return blockToInsertTo, childOfBlockToInsertBefore
        """
        assert isinstance(node, ast.SympyAssignment)
        assert isinstance(node.parent, ast.Block)

        lastBlock = node.parent
        lastBlockChild = node
        element = node.parent
        prevElement = node
        while element:
            if isinstance(element, ast.Block):
                lastBlock = element
                lastBlockChild = prevElement

            if isinstance(element, ast.Conditional):
                criticalSymbols = element.conditionExpr.atoms(sp.Symbol)
            else:
                criticalSymbols = element.symbolsDefined
            if node.undefinedSymbols.intersection(criticalSymbols):
                break
            prevElement = element
            element = element.parent
        return lastBlock, lastBlockChild

    def checkIfAssignmentAlreadyInBlock(assignment, targetBlock):
        for arg in targetBlock.args:
            if type(arg) is not ast.SympyAssignment:
                continue
            if arg.lhs == assignment.lhs:
                return arg
        return None

    def getBlocks(node, resultList):
        if isinstance(node, ast.Block):
            resultList.insert(0, node)
        if isinstance(node, ast.Node):
            for a in node.args:
                getBlocks(a, resultList)

    allBlocks = []
    getBlocks(astNode, allBlocks)
    for block in allBlocks:
        children = block.takeChildNodes()
        for child in children:
            if not isinstance(child, ast.SympyAssignment):
                block.append(child)
            else:
                target, childToInsertBefore = findBlockToMoveTo(child)
                if target == block:     # movement not possible
                    target.append(child)
                else:
                    existingAssignment = checkIfAssignmentAlreadyInBlock(child, target)
                    if not existingAssignment:
                        target.insertBefore(child, childToInsertBefore)
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

    symbolsWithTemporaryArray = OrderedDict()
    assignmentMap = OrderedDict((a.lhs, a) for a in innerLoop.body.args)

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
                newTs = TypedSymbol(symbol.name, PointerType(symbol.dtype))
                symbolsWithTemporaryArray[symbol] = IndexedBase(newTs, shape=(1,))[innerLoop.loopCounterSymbol]

        assignmentGroup = []
        for assignment in innerLoop.body.args:
            if assignment.lhs in symbolsResolved:
                newRhs = assignment.rhs.subs(symbolsWithTemporaryArray.items())
                if type(assignment.lhs) is not Field.Access and assignment.lhs in symbolGroup:
                    assert type(assignment.lhs) is TypedSymbol
                    newTs = TypedSymbol(assignment.lhs.name, PointerType(assignment.lhs.dtype))
                    newLhs = IndexedBase(newTs, shape=(1,))[innerLoop.loopCounterSymbol]
                else:
                    newLhs = assignment.lhs
                assignmentGroup.append(ast.SympyAssignment(newLhs, newRhs))
        assignmentGroups.append(assignmentGroup)

    newLoops = [innerLoop.newLoopWithDifferentBody(ast.Block(group)) for group in assignmentGroups]
    innerLoop.parent.replace(innerLoop, ast.Block(newLoops))

    for tmpArray in symbolsWithTemporaryArray:
        tmpArrayPointer = TypedSymbol(tmpArray.name, PointerType(tmpArray.dtype))
        outerLoop.parent.insertFront(ast.TemporaryMemoryAllocation(tmpArrayPointer, innerLoop.stop))
        outerLoop.parent.append(ast.TemporaryMemoryFree(tmpArrayPointer))


def cutLoop(loopNode, cuttingPoints):
    """Cuts loop at given cutting points, that means one loop is transformed into len(cuttingPoints)+1 new loops
    that range from  oldBegin to cuttingPoint[1], ..., cuttingPoint[-1] to oldEnd"""
    if loopNode.step != 1:
        raise NotImplementedError("Can only split loops that have a step of 1")
    newLoops = []
    newStart = loopNode.start
    cuttingPoints = list(cuttingPoints) + [loopNode.stop]
    for newEnd in cuttingPoints:
        if newEnd - newStart == 1:
            newBody = deepcopy(loopNode.body)
            newBody.subs({loopNode.loopCounterSymbol: newStart})
            newLoops.append(newBody)
        else:
            newLoop = ast.LoopOverCoordinate(deepcopy(loopNode.body), loopNode.coordinateToLoopOver,
                                             newStart, newEnd, loopNode.step)
            newLoops.append(newLoop)
        newStart = newEnd
    loopNode.parent.replace(loopNode, newLoops)


def isConditionNecessary(condition, preCondition, symbol):
    """
    Determines if a logical condition of a single variable is already contained in a stronger preCondition
    so if from preCondition follows that condition is always true, then this condition is not necessary
    :param condition: sympy relational of one variable
    :param preCondition: logical expression that is known to be true
    :param symbol: the single symbol of interest
    :return: returns  not (preCondition => condition) where "=>" is logical implication
    """
    from sympy.solvers.inequalities import reduce_rational_inequalities
    from sympy.logic.boolalg import to_dnf

    def toDnfList(expr):
        result = to_dnf(expr)
        if isinstance(result, sp.Or):
            return [orTerm.args for orTerm in result.args]
        elif isinstance(result, sp.And):
            return [result.args]
        else:
            return result

    t1 = reduce_rational_inequalities(toDnfList(sp.And(condition, preCondition)), symbol)
    t2 = reduce_rational_inequalities(toDnfList(preCondition), symbol)
    return t1 != t2


def simplifyBooleanExpression(expr, singleVariableRanges):
    """Simplification of boolean expression using known ranges of variables
    The singleVariableRanges parameter is a dict mapping a variable name to a sympy logical expression that
    contains only this variable and defines a range for it. For example with a being a symbol
    { a: sp.And(a >=0, a < 10) }
    """
    from sympy.core.relational import Relational
    from sympy.logic.boolalg import to_dnf

    expr = to_dnf(expr)

    def visit(e):
        if isinstance(e, Relational):
            symbols = e.atoms(sp.Symbol)
            if len(symbols) == 1:
                symbol = symbols.pop()
                if symbol in singleVariableRanges:
                    if not isConditionNecessary(e, singleVariableRanges[symbol], symbol):
                        return sp.true
            return e
        else:
            newArgs = [visit(a) for a in e.args]
            return e.func(*newArgs) if newArgs else e

    return visit(expr)


def simplifyConditionals(node, loopConditionals={}):
    """Simplifies/Removes conditions inside loops that depend on the loop counter."""
    if isinstance(node, ast.LoopOverCoordinate):
        ctrSym = node.loopCounterSymbol
        loopConditionals[ctrSym] = sp.And(ctrSym >= node.start, ctrSym < node.stop)
        simplifyConditionals(node.body)
        del loopConditionals[ctrSym]
    elif isinstance(node, ast.Conditional):
        node.conditionExpr = simplifyBooleanExpression(node.conditionExpr, loopConditionals)
        simplifyConditionals(node.trueBlock)
        if node.falseBlock:
            simplifyConditionals(node.falseBlock)
        if node.conditionExpr == sp.true:
            node.parent.replace(node, [node.trueBlock])
        if node.conditionExpr == sp.false:
            node.parent.replace(node, [node.falseBlock] if node.falseBlock else [])
    elif isinstance(node, ast.Block):
        for a in list(node.args):
            simplifyConditionals(a)
    elif isinstance(node, ast.SympyAssignment):
        return node
    else:
        raise ValueError("Can not handle node", type(node))


def cleanupBlocks(node):
    """Curly Brace Removal: Removes empty blocks, and replaces blocks with a single child by its child """
    if isinstance(node, ast.SympyAssignment):
        return
    elif isinstance(node, ast.Block):
        for a in list(node.args):
            cleanupBlocks(a)
        if len(node.args) <= 1 and isinstance(node.parent, ast.Block):
            node.parent.replace(node, node.args)
            return
    else:
        for a in node.args:
            cleanupBlocks(a)


def symbolNameToVariableName(symbolName):
    """Replaces characters which are allowed in sympy symbol names but not in C/C++ variable names"""
    return symbolName.replace("^", "_")


def typeAllEquations(eqs, typeForSymbol):
    """
    Traverses AST and replaces every :class:`sympy.Symbol` by a :class:`pystencils.typedsymbol.TypedSymbol`.
    Additionally returns sets of all fields which are read/written

    :param eqs: list of equations
    :param typeForSymbol: dict mapping symbol names to types. Types are strings of C types like 'int' or 'double'
    :return: ``fieldsRead, fieldsWritten, typedEquations`` set of read fields, set of written fields, list of equations
               where symbols have been replaced by typed symbols
    """
    if isinstance(typeForSymbol, str) or not hasattr(typeForSymbol, '__getitem__'):
        typeForSymbol = typingFromSympyInspection(eqs, typeForSymbol)

    fieldsWritten = set()
    fieldsRead = set()

    def processRhs(term):
        """Replaces Symbols by:
            - TypedSymbol if symbol is not a field access
        """
        if isinstance(term, Field.Access):
            fieldsRead.add(term.field)
            return term
        elif isinstance(term, TypedSymbol):
            return term
        elif isinstance(term, sp.Symbol):
            return TypedSymbol(symbolNameToVariableName(term.name), typeForSymbol[term.name])
        else:
            newArgs = [processRhs(arg) for arg in term.args]
            return term.func(*newArgs) if newArgs else term

    def processLhs(term):
        """Replaces symbol by TypedSymbol and adds field to fieldsWriten"""
        if isinstance(term, Field.Access):
            fieldsWritten.add(term.field)
            return term
        elif isinstance(term, TypedSymbol):
            return term
        elif isinstance(term, sp.Symbol):
            return TypedSymbol(term.name, typeForSymbol[term.name])
        else:
            assert False, "Expected a symbol as left-hand-side"

    def visit(object):
        if isinstance(object, list) or isinstance(object, tuple):
            return [visit(e) for e in object]
        if isinstance(object, sp.Eq) or isinstance(object, ast.SympyAssignment):
            newLhs = processLhs(object.lhs)
            newRhs = processRhs(object.rhs)
            return ast.SympyAssignment(newLhs, newRhs)
        elif isinstance(object, ast.Conditional):
            falseBlock = None if object.falseBlock is None else visit(object.falseBlock)
            return ast.Conditional(processRhs(object.conditionExpr),
                                   trueBlock=visit(object.trueBlock), falseBlock=falseBlock)
        elif isinstance(object, ast.Block):
            return ast.Block([visit(e) for e in object.args])
        else:
            return object

    typedEquations = visit(eqs)

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
        if isinstance(eq, ast.Node):
            continue
        # problematic case here is when rhs is a symbol: then it is impossible to decide here without
        # further information what type the left hand side is - default fallback is the dict value then
        if isinstance(eq.rhs, Boolean) and not isinstance(eq.rhs, sp.Symbol):
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
    :return: list of coordinate ids, where the first list entry should be the outermost loop
    """
    assert len(fields) > 0
    refField = next(iter(fields))
    for field in fields:
        if field.spatialDimensions != refField.spatialDimensions:
            raise ValueError("All fields have to have the same number of spatial dimensions. Spatial field dimensions: "
                             + str({f.name: f.spatialShape for f in fields}))

    layouts = set([field.layout for field in fields])
    if len(layouts) > 1:
        raise ValueError("Due to different layout of the fields no optimal loop ordering exists " +
                         str({f.name: f.layout for f in fields}))
    layout = list(layouts)[0]
    return list(layout)


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
    return reversed(result)


def get_type(node):
    if isinstance(node, ast.Indexed):
        return node.args[0].dtype
    elif isinstance(node, ast.Node):
        return node.dtype
    # TODO sp.NumberSymbol
    elif isinstance(node, sp.Number):
        if isinstance(node, sp.Float):
            return createType('double')
        elif isinstance(node, sp.Integer):
            return createType('int')
        else:
            raise NotImplemented('Not yet supported: %s %s' % (node, type(node)))
    else:
        raise NotImplemented('Not yet supported: %s %s' % (node, type(node)))


def insert_casts(node):
    """
    Inserts casts and dtype whpere needed
    :param node: ast which should be traversed
    :return: node
    """
    def conversion(args):
        target = args[0]
        if isinstance(target.dtype, PointerType):
            # Pointer arithmetic
            for arg in args[1:]:
                # Check validness
                if not arg.dtype.is_int() and not arg.dtype.is_uint():
                    raise ValueError("Impossible pointer arithmetic", target, arg)
            pointer = ast.PointerArithmetic(ast.Add(args[1:]), target)
            return [pointer]

        else:
            for i in range(len(args)):
                if args[i].dtype != target.dtype:
                    args[i] = ast.Conversion(args[i], target.dtype, node)
            return args

    for arg in node.args:
        insert_casts(arg)
    if isinstance(node, ast.Indexed):
        #TODO revmove this
        pass
    elif isinstance(node, ast.Expr):
        #print(node, node.args)
        #print([type(arg) for arg in node.args])
        #print([arg.dtype for arg in node.args])
        args = sorted((arg for arg in node.args), key=attrgetter('dtype'))
        target = args[0]
        node.args = conversion(args)
        node.dtype = target.dtype
        #print(node.dtype)
        #print(node)
    elif isinstance(node, ast.SympyAssignment):
        if node.lhs.dtype != node.rhs.dtype:
            node.replace(node.rhs, ast.Conversion(node.rhs, node.lhs.dtype))
    elif isinstance(node, ast.LoopOverCoordinate):
        pass
    return node


def desympy_ast(node):
    """
    Remove Sympy Expressions, which have more then one argument.
    This is necessary for further changes in the tree.
    :param node: ast which should be traversed. Only node's children will be modified.
    :return: (modified) node
    """
    if node.args is None:
        return node
    for i in range(len(node.args)):
        arg = node.args[i]
        if isinstance(arg, sp.Add):
            node.replace(arg, ast.Add(arg.args, node))
        elif isinstance(arg, sp.Number):
            node.replace(arg, ast.Number(arg, node))
        elif isinstance(arg, sp.Mul):
            node.replace(arg, ast.Mul(arg.args, node))
        elif isinstance(arg, sp.Pow):
            node.replace(arg, ast.Pow(arg.args, node))
        elif isinstance(arg, sp.tensor.Indexed) or isinstance(arg, sp.tensor.indexed.Indexed):
            node.replace(arg, ast.Indexed(arg.args, arg.base, node))
        elif isinstance(arg,  sp.tensor.IndexedBase):
            node.replace(arg, arg.label)
        #elif isinstance(arg, sp.containers.Tuple):
        #
        else:
            #print('Not transforming:', type(arg), arg)
            pass
    for arg in node.args:
        desympy_ast(arg)
    return node


def check_dtype(node):
    if isinstance(node, ast.KernelFunction):
        pass
    elif isinstance(node, ast.Block):
        pass
    elif isinstance(node, ast.LoopOverCoordinate):
        pass
    elif isinstance(node, ast.SympyAssignment):
        pass
    else:
        #print(node)
        #print(node.dtype)
        pass
    for arg in node.args:
        check_dtype(arg)

