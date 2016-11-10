import sympy as sp
from sympy.tensor import IndexedBase, Indexed
from pystencils.field import Field
from pystencils.typedsymbol import TypedSymbol


class Node(object):
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

    def parents(self):
        return None

    def children(self):
        return None


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

        def __repr__(self):
            return '<{0} {1}>'.format(self.dtype, self.name)

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

    def children(self):
        yield self.body

    def __repr__(self):
        self._updateParameters()
        return '{0} {1}({2})\n{3}'.format(type(self).__name__, self.functionName, self.parameters, self.body)


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

    def insertBefore(self, newNode, insertBefore):
        idx = self._nodes.index(insertBefore)
        self._nodes.insert(idx, newNode)

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

    def children(self):
        yield self._nodes

    def __repr__(self):
        return ''.join('\t{!r}\n'.format(node) for node in self._nodes)


class PragmaBlock(Block):
    def __init__(self, pragmaLine, listOfNodes):
        super(PragmaBlock, self).__init__(listOfNodes)
        self.pragmaLine = pragmaLine


class LoopOverCoordinate(Node):
    LOOP_COUNTER_NAME_PREFIX = "ctr"

    def __init__(self, body, coordinateToLoopOver, start, stop, step=1):
        self._body = body
        self._coordinateToLoopOver = coordinateToLoopOver
        self._begin = start
        self._end = stop
        self._increment = step
        self._body.parent = self
        self.prefixLines = []

    def newLoopWithDifferentBody(self, newBody):
        result = LoopOverCoordinate(newBody, self._coordinateToLoopOver, self._begin, self._end, self._increment)
        result.prefixLines = self.prefixLines
        return result

    @property
    def args(self):
        result = [self._body]
        for e in [self._begin, self._end, self._increment]:
            if hasattr(e, "args"):
                result.append(e)
        return result

    @property
    def body(self):
        return self._body

    @property
    def start(self):
        return self._begin

    @property
    def stop(self):
        return self._end

    @property
    def step(self):
        return self._increment

    @property
    def coordinateToLoopOver(self):
        return self._coordinateToLoopOver

    @property
    def symbolsDefined(self):
        result = self._body.symbolsDefined
        result.add(self.loopCounterSymbol)
        return result

    @staticmethod
    def getLoopCounterName(coordinateToLoopOver):
        return "%s_%s" % (LoopOverCoordinate.LOOP_COUNTER_NAME_PREFIX, coordinateToLoopOver)

    @property
    def loopCounterName(self):
        return LoopOverCoordinate.getLoopCounterName(self.coordinateToLoopOver)

    @staticmethod
    def getLoopCounterSymbol(coordinateToLoopOver):
        return TypedSymbol(LoopOverCoordinate.getLoopCounterName(coordinateToLoopOver), "int")

    @property
    def loopCounterSymbol(self):
        return LoopOverCoordinate.getLoopCounterSymbol(self.coordinateToLoopOver)

    @property
    def symbolsRead(self):
        loopBoundSymbols = set()
        for possibleSymbol in [self._begin, self._end, self._increment]:
            if isinstance(possibleSymbol, Node) or isinstance(possibleSymbol, sp.Basic):
                loopBoundSymbols.update(possibleSymbol.atoms(sp.Symbol))
        result = self._body.symbolsRead.union(loopBoundSymbols)
        return result

    @property
    def isOutermostLoop(self):
        from pystencils.transformations import getNextParentOfType
        return getNextParentOfType(self, LoopOverCoordinate) is None

    @property
    def isInnermostLoop(self):
        return len(self.atoms(LoopOverCoordinate)) == 0

    @property
    def coordinateToLoopOver(self):
        return self._coordinateToLoopOver

    def children(self):
        return self.body

    def __repr__(self):
        return 'loop:{!s} {!s} in {!s}:{!s}:{!s}\n'.format(self.loopCounterName, self.coordinateToLoopOver, self.start,
                                                           self.stop, self.step) + '\t{!r}\n'.format(self.body)


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

