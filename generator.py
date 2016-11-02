from collections import defaultdict
import cgen as c
import sympy as sp
from sympy.logic.boolalg import Boolean
from sympy.utilities.codegen import CCodePrinter
from sympy.tensor import IndexedBase, Indexed
from pystencils.field import Field, offsetComponentToDirectionString
from pystencils.typedsymbol import TypedSymbol

COORDINATE_LOOP_COUNTER_NAME = "ctr"
FIELD_PTR_PREFIX = Field.PREFIX + "d_"


# --------------------------------------- Helper Functions -------------------------------------------------------------


class CodePrinter(CCodePrinter):
    def _print_Pow(self, expr):
        if expr.exp.is_integer and expr.exp.is_number and expr.exp > 0:
            return '(' + '*'.join(["(" + self._print(expr.base) + ")"] * expr.exp) + ')'
        else:
            return super(CodePrinter, self)._print_Pow(expr)

    def _print_Rational(self, expr):
        return str(expr.evalf().num)

    def _print_Equality(self, expr):
        return '((' + self._print(expr.lhs) + ") == (" + self._print(expr.rhs) + '))'

    def _print_Piecewise(self, expr):
        result = super(CodePrinter, self)._print_Piecewise(expr)
        return result.replace("\n", "")

codePrinter = CodePrinter()


class MyPOD(c.Declarator):
    def __init__(self, dtype, name):
        self.dtype = dtype
        self.name = name

    def get_decl_pair(self):
        return [self.dtype], self.name


def getNextParentOfType(node, parentType):
    parent = node.parent
    while parent is not None:
        if isinstance(parent, parentType):
            return parent
        parent = parent.parent
    return None


# --------------------------------------- AST Nodes  -------------------------------------------------------------------


class Node:
    def __init__(self, parent=None):
        self.parent = parent

    def args(self):
        return []

    def atoms(self, argType):
        result = set()
        for arg in self.args:
            if isinstance(arg, argType):
                result.add(arg)
            result.update(arg.atoms(argType))
        return result


class DebugNode(Node):
    def __init__(self, code, symbolsRead=[]):
        self._code = code
        self._symbolsRead = set(symbolsRead)

    @property
    def args(self):
        return []

    @property
    def symbolsDefined(self):
        return set()

    @property
    def symbolsRead(self):
        return self._symbolsRead

    def generateC(self):
        return c.LiteralLines(self._code)


class PrintNode(DebugNode):
    def __init__(self, symbolToPrint):
        code = '\nstd::cout << "%s  =  " << %s << std::endl; \n' % (symbolToPrint.name, symbolToPrint.name)
        super(PrintNode, self).__init__(code, [symbolToPrint])


class KernelFunction(Node):

    class Argument:
        def __init__(self, name, dtype):
            self.name = name
            self.dtype = dtype
            self.isFieldPtrArgument = False
            self.isFieldShapeArgument = False
            self.isFieldStrideArgument = False
            self.isFieldArgument = False
            self.fieldName = ""
            self.coordinate = None

            if name.startswith(FIELD_PTR_PREFIX):
                self.isFieldPtrArgument = True
                self.isFieldArgument = True
                self.fieldName = name[len(FIELD_PTR_PREFIX):]
            elif name.startswith(Field.SHAPE_PREFIX):
                self.isFieldShapeArgument = True
                self.isFieldArgument = True
                self.fieldName = name[len(Field.SHAPE_PREFIX):]
            elif name.startswith(Field.STRIDE_PREFIX):
                self.isFieldStrideArgument = True
                self.isFieldArgument = True
                self.fieldName = name[len(Field.STRIDE_PREFIX):]

    def __init__(self, body, functionName="kernel"):
        super(KernelFunction, self).__init__()
        self._body = body
        self._parameters = None
        self._functionName = functionName
        self._body.parent = self
        self.variablesToIgnore = set()
        self.qualifierPrefix = ""

    @property
    def symbolsDefined(self):
        return set()

    @property
    def symbolsRead(self):
        return set()

    @property
    def parameters(self):
        self._updateArguments()
        return self._parameters

    @property
    def body(self):
        return self._body

    @property
    def args(self):
        return [self._body]

    @property
    def functionName(self):
        return self._functionName

    def _updateArguments(self):
        undefinedSymbols = self._body.symbolsRead - self._body.symbolsDefined - self.variablesToIgnore
        self._parameters = [KernelFunction.Argument(s.name, s.dtype) for s in undefinedSymbols]
        self._parameters.sort(key=lambda l: (l.fieldName, l.isFieldPtrArgument, l.isFieldShapeArgument,
                                             l.isFieldStrideArgument, l.name),
                              reverse=True)

    def generateC(self):
        self._updateArguments()
        functionArguments = [MyPOD(s.dtype, s.name) for s in self._parameters]
        functionPOD = MyPOD(self.qualifierPrefix + "void", self._functionName, )
        funcDeclaration = c.FunctionDeclaration(functionPOD, functionArguments)
        return c.FunctionBody(funcDeclaration, self._body.generateC())


class Block(Node):
    def __init__(self, listOfNodes):
        super(Node, self).__init__()
        self._nodes = listOfNodes
        for n in self._nodes:
            n.parent = self

    @property
    def args(self):
        return self._nodes

    def insertFront(self, node):
        node.parent = self
        self._nodes.insert(0, node)

    def append(self, node):
        node.parent = self
        self._nodes.append(node)

    def generateC(self):
        return c.Block([e.generateC() for e in self.args])

    def takeChildNodes(self):
        tmp = self._nodes
        self._nodes = []
        return tmp

    def replace(self, child, replacements):
        idx = self._nodes.index(child)
        del self._nodes[idx]
        if type(replacements) is list:
            for e in replacements:
                e.parent = self
            self._nodes = self._nodes[:idx] + replacements + self._nodes[idx:]
        else:
            replacements.parent = self
            self._nodes.insert(idx, replacements)

    @property
    def symbolsDefined(self):
        result = set()
        for a in self.args:
            result.update(a.symbolsDefined)
        return result

    @property
    def symbolsRead(self):
        result = set()
        for a in self.args:
            result.update(a.symbolsRead)
        return result


class PragmaBlock(Block):
    def __init__(self, pragmaLine, listOfNodes):
        super(PragmaBlock, self).__init__(listOfNodes)
        self._pragmaLine = pragmaLine

    def generateC(self):
        class PragmaGenerable(c.Generable):
            def __init__(self, line, block):
                self._line = line
                self._block = block

            def generate(self):
                yield self._line
                for e in self._block.generate():
                    yield e

        return PragmaGenerable(self._pragmaLine, super(PragmaBlock, self).generateC())


class LoopOverCoordinate(Node):

    def __init__(self, body, coordinateToLoopOver, shape, increment=1, ghostLayers=1,
                 isInnermostLoop=False, isOutermostLoop=False):
        self._body = body
        self._coordinateToLoopOver = coordinateToLoopOver
        self._shape = shape
        self._increment = increment
        self._ghostLayers = ghostLayers
        self._body.parent = self
        self.prefixLines = []
        self._isInnermostLoop = isInnermostLoop
        self._isOutermostLoop = isOutermostLoop

    def newLoopWithDifferentBody(self, newBody):
        result = LoopOverCoordinate(newBody, self._coordinateToLoopOver, self._shape, self._increment,
                                    self._ghostLayers, self._isInnermostLoop, self._isOutermostLoop)
        result.prefixLines = self.prefixLines
        return result

    @property
    def args(self):
        result = [self._body]
        limit = self._shape[self._coordinateToLoopOver]
        if isinstance(limit, Node) or isinstance(limit, sp.Basic):
            result.append(limit)
        return result

    @property
    def body(self):
        return self._body

    @property
    def loopCounterName(self):
        return "%s_%s" % (COORDINATE_LOOP_COUNTER_NAME, self._coordinateToLoopOver)

    @property
    def coordinateToLoopOver(self):
        return self._coordinateToLoopOver

    @property
    def symbolsDefined(self):
        result = self._body.symbolsDefined
        result.add(self.loopCounterSymbol)
        return result

    @property
    def loopCounterSymbol(self):
        return TypedSymbol(self.loopCounterName, "int")

    @property
    def symbolsRead(self):
        result = self._body.symbolsRead
        limit = self._shape[self._coordinateToLoopOver]
        if isinstance(limit, sp.Basic):
            result.update(limit.atoms(sp.Symbol))
        return result

    @property
    def isOutermostLoop(self):
        return self._isOutermostLoop

    @property
    def isInnermostLoop(self):
        return self._isInnermostLoop

    @property
    def coordinateToLoopOver(self):
        return self._coordinateToLoopOver

    @property
    def iterationRegionWithGhostLayer(self):
        return self._shape[self._coordinateToLoopOver]

    def generateC(self):
        coord = self._coordinateToLoopOver
        end = self._shape[coord] - self._ghostLayers

        counterVar = self.loopCounterName

        class LoopWithOptionalPrefix(c.CustomLoop):
            def __init__(self, intro_line, body, prefixLines=[]):
                super(LoopWithOptionalPrefix, self).__init__(intro_line, body)
                self.prefixLines = prefixLines

            def generate(self):
                for l in self.prefixLines:
                    yield l

                for e in super(LoopWithOptionalPrefix, self).generate():
                    yield e

        start = "int %s = %d" % (counterVar, self._ghostLayers)
        condition = "%s < %s" % (counterVar, codePrinter.doprint(end))
        update = "++%s" % (counterVar,)
        loopStr = "for (%s; %s; %s)" % (start, condition, update)
        return LoopWithOptionalPrefix(loopStr, self._body.generateC(), prefixLines=self.prefixLines)


class SympyAssignment(Node):

    def __init__(self, lhsSymbol, rhsTerm, isConst=True):
        self._lhsSymbol = lhsSymbol
        self.rhs = rhsTerm
        self._isDeclaration = True
        if isinstance(self._lhsSymbol, Field.Access) or isinstance(self._lhsSymbol, IndexedBase):
            self._isDeclaration = False
        self._isConst = isConst

    @property
    def lhs(self):
        return self._lhsSymbol

    @lhs.setter
    def lhs(self, newValue):
        self._lhsSymbol = newValue
        self._isDeclaration = True
        if isinstance(self._lhsSymbol, Field.Access) or isinstance(self._lhsSymbol, Indexed):
            self._isDeclaration = False

    @property
    def args(self):
        return [self._lhsSymbol, self.rhs]

    @property
    def symbolsDefined(self):
        if not self._isDeclaration:
            return set()
        return set([self._lhsSymbol])

    @property
    def symbolsRead(self):
        result = self.rhs.atoms(sp.Symbol)
        result.update(self._lhsSymbol.atoms(sp.Symbol))
        return result

    @property
    def isConst(self):
        return self._isConst

    def __repr__(self):
        return repr(self.lhs) + " = " + repr(self.rhs)

    def generateC(self):
        dtype = ""
        if hasattr(self._lhsSymbol, 'dtype') and self._isDeclaration:
            if self._isConst:
                dtype = "const " + self._lhsSymbol.dtype + " "
            else:
                dtype = self._lhsSymbol.dtype + " "

        return c.Assign(dtype + codePrinter.doprint(self._lhsSymbol),
                        codePrinter.doprint(self.rhs))


class CustomCppCode(Node):
    def __init__(self, code, symbolsRead, symbolsDefined):
        self._code = "\n" + code
        self._symbolsRead = set(symbolsRead)
        self._symbolsDefined = set(symbolsDefined)

    @property
    def args(self):
        return []

    @property
    def symbolsDefined(self):
        return self._symbolsDefined

    @property
    def symbolsRead(self):
        return self._symbolsRead

    def generateC(self):
        return c.LiteralLines(self._code)


class TemporaryArrayDefinition(Node):
    def __init__(self, typedSymbol, size):
        self._symbol = typedSymbol
        self._size = size

    @property
    def symbolsDefined(self):
        return set([self._symbol])

    @property
    def symbolsRead(self):
        return set()

    def generateC(self):
        return c.Assign(self._symbol.dtype + " * " + codePrinter.doprint(self._symbol),
                        "new %s[%s]" % (self._symbol.dtype, codePrinter.doprint(self._size)))

    @property
    def args(self):
        return [self._symbol]


class TemporaryArrayDelete(Node):
    def __init__(self, typedSymbol):
        self._symbol = typedSymbol

    @property
    def symbolsDefined(self):
        return set()

    @property
    def symbolsRead(self):
        return set()

    def generateC(self):
        return c.Statement("delete [] %s" % (codePrinter.doprint(self._symbol),))

    @property
    def args(self):
        return []


# --------------------------------------- Factory Functions ------------------------------------------------------------


def getOptimalLoopOrdering(fields):
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


def makeLoopOverDomain(body, functionName):
    """
    :param body: list of nodes
    :param functionName: name of generated C function
    :return: LoopOverCoordinate instance with nested loops, ordered according to field layouts
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
        newLoop = LoopOverCoordinate(currentBody, loopCoordinate, shape, 1, requiredGhostLayers,
                                     isInnermostLoop=(i == 0), isOutermostLoop=(i == len(loopOrder) - 1))
        lastLoop = newLoop
        currentBody = Block([lastLoop])
    return KernelFunction(currentBody, functionName)


# --------------------------------------- Transformations --------------------------------------------------------------

def createIntermediateBasePointer(fieldAccess, coordinates, previousPtr):
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
    Allowed specifications:
    "spatialInner<int>" spatialInner0 is the innermost loop coordinate, spatialInner1 the loop enclosing the innermost
    "spatialOuter<int>" spatialOuter0 is the outermost loop
    "index<int>": index coordinate
    "<int>": specifying directly the coordinate
    :param basePointerSpecification: nested list with above specifications
    :param loopOrder: list with ordering of loops from inner to outer
    :param field:
    :return:
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


def getLoopHierarchy(block):
    result = []
    node = block
    while node is not None:
        node = getNextParentOfType(node, LoopOverCoordinate)
        if node:
            result.append(node.coordinateToLoopOver)
    return result


def resolveFieldAccesses(ast, fieldToBasePointerInfo={}, fieldToFixedCoordinates={}):
    """Substitutes FieldAccess nodes by array indexing"""

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

            fieldPtr = TypedSymbol("%s%s" % (FIELD_PTR_PREFIX, field.name), dtype)

            lastPointer = fieldPtr

            def createCoordinateDict(group):
                coordDict = {}
                for e in group:
                    if e < field.spatialDimensions:
                        if field.name in fieldToFixedCoordinates:
                            coordDict[e] = fieldToFixedCoordinates[field.name][e]
                        else:
                            coordDict[e] = TypedSymbol("%s_%d" % (COORDINATE_LOOP_COUNTER_NAME, e), "int")
                    else:
                        coordDict[e] = fieldAccess.index[e-field.spatialDimensions]
                return coordDict

            for group in reversed(basePointerInfo[1:]):
                coordDict = createCoordinateDict(group)
                newPtr, offset = createIntermediateBasePointer(fieldAccess, coordDict, lastPointer)
                if newPtr not in enclosingBlock.symbolsDefined:
                    enclosingBlock.insertFront(SympyAssignment(newPtr, lastPointer + offset, isConst=False))
                lastPointer = newPtr

            _, offset = createIntermediateBasePointer(fieldAccess, createCoordinateDict(basePointerInfo[0]), lastPointer)
            baseArr = IndexedBase(lastPointer, shape=(1,))
            return baseArr[offset]
        else:
            newArgs = [visitSympyExpr(e, enclosingBlock) for e in expr.args]
            kwargs = {'evaluate': False} if type(expr) is sp.Add or type(expr) is sp.Mul else {}
            return expr.func(*newArgs, **kwargs) if newArgs else expr

    def visitNode(subAst):
        if isinstance(subAst, SympyAssignment):
            enclosingBlock = subAst.parent
            assert type(enclosingBlock) is Block
            subAst.lhs = visitSympyExpr(subAst.lhs, enclosingBlock)
            subAst.rhs = visitSympyExpr(subAst.rhs, enclosingBlock)
        else:
            for i, a in enumerate(subAst.args):
                visitNode(a)

    return visitNode(ast)


def moveConstantsBeforeLoop(ast):

    def findBlockToMoveTo(node):
        """Traverses parents of node as long as the symbols are independent and returns a (parent) block
        the assignment can be safely moved to
        :param node: SympyAssignment inside a Block"""
        assert isinstance(node, SympyAssignment)
        assert isinstance(node.parent, Block)

        lastBlock = node.parent
        element = node.parent
        while element:
            if isinstance(element, Block):
                lastBlock = element
            if node.symbolsRead.intersection(element.symbolsDefined):
                break
            element = element.parent
        return lastBlock

    def checkIfAssignmentAlreadyInBlock(assignment, targetBlock):
        for arg in targetBlock.args:
            if type(arg) is not SympyAssignment:
                continue
            if arg.lhs == assignment.lhs:
                return arg
        return None

    for block in ast.atoms(Block):
        children = block.takeChildNodes()
        for child in children:
            if not isinstance(child, SympyAssignment):
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


def splitInnerLoop(ast, symbolGroups):
    allLoops = ast.atoms(LoopOverCoordinate)
    innerLoop = [l for l in allLoops if l.isInnermostLoop]
    assert len(innerLoop) == 1, "Error in AST: multiple innermost loops. Was split transformation already called?"
    innerLoop = innerLoop[0]
    assert type(innerLoop.body) is Block
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
                assignmentGroup.append(SympyAssignment(newLhs, newRhs))
        assignmentGroups.append(assignmentGroup)

    newLoops = [innerLoop.newLoopWithDifferentBody(Block(group)) for group in assignmentGroups]
    innerLoop.parent.replace(innerLoop, newLoops)

    for tmpArray in symbolsWithTemporaryArray:
        outerLoop.parent.insertFront(TemporaryArrayDefinition(tmpArray, innerLoop.iterationRegionWithGhostLayer))
        outerLoop.parent.append(TemporaryArrayDelete(tmpArray))


# ------------------------------------- Main ---------------------------------------------------------------------------


def extractCommonSubexpressions(equations):
    """Uses sympy to find common subexpressions in equations and returns
    them in a topologically sorted order, ready for evaluation"""
    replacements, newEq = sp.cse(equations)
    replacementEqs = [sp.Eq(*r) for r in replacements]
    equations = replacementEqs + newEq
    topologicallySortedPairs = sp.cse_main.reps_toposort([[e.lhs, e.rhs] for e in equations])
    equations = [sp.Eq(*a) for a in topologicallySortedPairs]
    return equations


def addOpenMP(ast):
    assert type(ast) is KernelFunction
    body = ast.body
    wrapperBlock = PragmaBlock('#pragma omp parallel', body.takeChildNodes())
    body.append(wrapperBlock)

    outerLoops = [l for l in body.atoms(LoopOverCoordinate) if l.isOutermostLoop]
    assert outerLoops, "No outer loop found"
    assert len(outerLoops) <= 1, "More than one outer loop found. Which one should be parallelized?"
    outerLoops[0].prefixLines.append("#pragma omp for schedule(static)")


def typeAllEquations(eqs, typeForSymbol):
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
            typedEquations.append(SympyAssignment(newLhs, newRhs))
        else:
            assert isinstance(eq, Node), "Only equations and ast nodes are allowed in input"
            typedEquations.append(eq)

    typedEquations = typedEquations

    return fieldsRead, fieldsWritten, typedEquations


def typingFromSympyInspection(eqs, defaultType="double"):
    result = defaultdict(lambda: defaultType)
    for eq in eqs:
        if isinstance(eq.rhs, Boolean):
            result[eq.lhs.name] = "bool"
    return result


def createKernel(listOfEquations, functionName="kernel", typeForSymbol=None, splitGroups=[]):
    if not typeForSymbol:
        typeForSymbol = typingFromSympyInspection(listOfEquations, "double")

    def typeSymbol(term):
        if isinstance(term, Field.Access) or isinstance(term, TypedSymbol):
            return term
        elif isinstance(term, sp.Symbol):
            return TypedSymbol(term.name, typeForSymbol[term.name])
        else:
            raise ValueError("Term has to be field access or symbol")

    fieldsRead, fieldsWritten, assignments = typeAllEquations(listOfEquations, typeForSymbol)
    allFields = fieldsRead.union(fieldsWritten)

    for field in allFields:
        field.setReadOnly(False)
    for field in fieldsRead - fieldsWritten:
        field.setReadOnly()

    body = Block(assignments)
    code = makeLoopOverDomain(body, functionName)

    if splitGroups:
        typedSplitGroups = [[typeSymbol(s) for s in splitGroup] for splitGroup in splitGroups]
        splitInnerLoop(code, typedSplitGroups)

    loopOrder = getOptimalLoopOrdering(allFields)

    basePointerInfo = [['spatialInner0'], ['spatialInner1']]
    basePointerInfos = {f.name: parseBasePointerInfo(basePointerInfo, loopOrder, f) for f in allFields}

    resolveFieldAccesses(code, fieldToBasePointerInfo=basePointerInfos)
    moveConstantsBeforeLoop(code)
    addOpenMP(code)

    return code


if __name__ == "__main__":
    f = Field.createGeneric('f', 3, indexDimensions=1)
    pointerSpec = [['spatialInner0']]
    parseBasePointerInfo(pointerSpec, [0, 1, 2], f)