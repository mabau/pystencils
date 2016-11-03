import sympy as sp
from sympy.tensor import IndexedBase, Indexed
from pystencils.field import Field
from pystencils.typedsymbol import TypedSymbol


class Node:
    """Base class for all AST nodes"""

    def __init__(self, parent=None):
        self.parent = parent

    def args(self):
        """Returns all arguments/children of this node"""
        return []

    @property
    def symbolsDefined(self):
        """Set of symbols which are defined in this node or its children"""
        return set()

    @property
    def symbolsRead(self):
        """Set of symbols which are accessed/read in this node or its children"""
        return set()

    def atoms(self, argType):
        """
        Returns a set of all children which are an instance of the given argType
        """
        result = set()
        for arg in self.args:
            if isinstance(arg, argType):
                result.add(arg)
            result.update(arg.atoms(argType))
        return result


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

            if name.startswith(Field.DATA_PREFIX):
                self.isFieldPtrArgument = True
                self.isFieldArgument = True
                self.fieldName = name[len(Field.DATA_PREFIX):]
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

    @property
    def symbolsDefined(self):
        return set()

    @property
    def symbolsRead(self):
        return set()

    @property
    def parameters(self):
        self._updateParameters()
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

    def _updateParameters(self):
        undefinedSymbols = self._body.symbolsRead - self._body.symbolsDefined - self.variablesToIgnore
        self._parameters = [KernelFunction.Argument(s.name, s.dtype) for s in undefinedSymbols]
        self._parameters.sort(key=lambda l: (l.fieldName, l.isFieldPtrArgument, l.isFieldShapeArgument,
                                             l.isFieldStrideArgument, l.name),
                              reverse=True)


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
        self.pragmaLine = pragmaLine


class LoopOverCoordinate(Node):
    LOOP_COUNTER_NAME_PREFIX = "ctr"

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
    def iterationEnd(self):
        return self._shape[self.coordinateToLoopOver] - self.ghostLayers

    @property
    def coordinateToLoopOver(self):
        return self._coordinateToLoopOver

    @property
    def symbolsDefined(self):
        result = self._body.symbolsDefined
        result.add(self.loopCounterSymbol)
        return result

    @property
    def loopCounterName(self):
        return "%s_%s" % (LoopOverCoordinate.LOOP_COUNTER_NAME_PREFIX, self._coordinateToLoopOver)

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

    @property
    def ghostLayers(self):
        return self._ghostLayers


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
    def isDeclaration(self):
        return self._isDeclaration

    @property
    def isConst(self):
        return self._isConst

    def __repr__(self):
        return repr(self.lhs) + " = " + repr(self.rhs)


class TemporaryMemoryAllocation(Node):
    def __init__(self, typedSymbol, size):
        self.symbol = typedSymbol
        self.size = size

    @property
    def symbolsDefined(self):
        return set([self._symbol])

    @property
    def symbolsRead(self):
        return set()

    @property
    def args(self):
        return [self._symbol]


class TemporaryMemoryFree(Node):
    def __init__(self, typedSymbol):
        self._symbol = typedSymbol

    @property
    def symbolsDefined(self):
        return set()

    @property
    def symbolsRead(self):
        return set()

    @property
    def args(self):
        return []

