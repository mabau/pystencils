import sympy as sp
from sympy.tensor import IndexedBase
from pystencils.field import Field
from pystencils.data_types import TypedSymbol, createType, get_type_from_sympy, createTypeFromString, castFunc
from pystencils.sympyextensions import fastSubs


class ResolvedFieldAccess(sp.Indexed):
    def __new__(cls, base, linearizedIndex, field, offsets, idxCoordinateValues):
        if not isinstance(base, IndexedBase):
            base = IndexedBase(base, shape=(1,))
        obj = super(ResolvedFieldAccess, cls).__new__(cls, base, linearizedIndex)
        obj.field = field
        obj.offsets = offsets
        obj.idxCoordinateValues = idxCoordinateValues
        return obj

    def _eval_subs(self, old, new):
        return ResolvedFieldAccess(self.args[0],
                                   self.args[1].subs(old, new),
                                   self.field, self.offsets, self.idxCoordinateValues)

    def _hashable_content(self):
        superClassContents = super(ResolvedFieldAccess, self)._hashable_content()
        return superClassContents + tuple(self.offsets) + (repr(self.idxCoordinateValues), hash(self.field))

    @property
    def typedSymbol(self):
        return self.base.label

    def __str__(self):
        top = super(ResolvedFieldAccess, self).__str__()
        return "%s (%s)" % (top, self.typedSymbol.dtype)

    def __getnewargs__(self):
        return self.base, self.indices[0], self.field, self.offsets, self.idxCoordinateValues


class Node(object):
    """Base class for all AST nodes"""

    def __init__(self, parent=None):
        self.parent = parent

    def args(self):
        """Returns all arguments/children of this node"""
        return []

    @property
    def symbolsDefined(self):
        """Set of symbols which are defined by this node. """
        return set()

    @property
    def undefinedSymbols(self):
        """Symbols which are used but are not defined inside this node"""
        raise NotImplementedError()

    def subs(self, *args, **kwargs):
        """Inplace! substitute, similar to sympys but modifies ast and returns None"""
        for a in self.args:
            a.subs(*args, **kwargs)

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


class Conditional(Node):
    """Conditional"""
    def __init__(self, conditionExpr, trueBlock, falseBlock=None):
        """
        Create a new conditional node

        :param conditionExpr: sympy relational expression
        :param trueBlock: block which is run if conditional is true
        :param falseBlock: block which is run if conditional is false, or None if not needed
        """
        assert conditionExpr.is_Boolean or conditionExpr.is_Relational
        self.conditionExpr = conditionExpr
        self.trueBlock = trueBlock
        self.falseBlock = falseBlock

    @property
    def args(self):
        result = [self.conditionExpr, self.trueBlock]
        if self.falseBlock:
            result.append(self.falseBlock)
        return result

    @property
    def symbolsDefined(self):
        return set()

    @property
    def undefinedSymbols(self):
        result = self.trueBlock.undefinedSymbols
        if self.falseBlock:
            result = result.update(self.falseBlock.undefinedSymbols)
        result.update(self.conditionExpr.atoms(sp.Symbol))
        return result

    def __str__(self):
        return 'if:({!s}) '.format(self.conditionExpr)

    def __repr__(self):
        return 'if:({!r}) '.format(self.conditionExpr)


class KernelFunction(Node):

    class Argument:
        def __init__(self, name, dtype, symbol, kernelFunctionNode):
            from pystencils.transformations import symbolNameToVariableName
            self.name = name
            self.dtype = dtype
            self.isFieldPtrArgument = False
            self.isFieldShapeArgument = False
            self.isFieldStrideArgument = False
            self.isFieldArgument = False
            self.fieldName = ""
            self.coordinate = None
            self.symbol = symbol

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

            self.field = None
            if self.isFieldArgument:
                fieldMap = {symbolNameToVariableName(f.name): f for f in kernelFunctionNode.fieldsAccessed}
                self.field = fieldMap[self.fieldName]

        def __lt__(self, other):
            def score(l):
                if l.isFieldPtrArgument:
                    return -4
                elif l.isFieldShapeArgument:
                    return -3
                elif l.isFieldStrideArgument:
                    return -2
                return 0

            if score(self) < score(other):
                return True
            elif score(self) == score(other):
                return self.name < other.name
            else:
                return False

        def __repr__(self):
            return '<{0} {1}>'.format(self.dtype, self.name)

    def __init__(self, body, ghostLayers=None, functionName="kernel"):
        super(KernelFunction, self).__init__()
        self._body = body
        body.parent = self
        self._parameters = None
        self.functionName = functionName
        self._body.parent = self
        self.ghostLayers = ghostLayers
        # these variables are assumed to be global, so no automatic parameter is generated for them
        self.globalVariables = set()

    @property
    def symbolsDefined(self):
        return set()

    @property
    def undefinedSymbols(self):
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
    def fieldsAccessed(self):
        """Set of Field instances: fields which are accessed inside this kernel function"""
        return set(o.field for o in self.atoms(ResolvedFieldAccess))

    def _updateParameters(self):
        undefinedSymbols = self._body.undefinedSymbols - self.globalVariables
        self._parameters = [KernelFunction.Argument(s.name, s.dtype, s, self) for s in undefinedSymbols]

        self._parameters.sort()

    def __str__(self):
        self._updateParameters()
        return '{0} {1}({2})\n{3}'.format(type(self).__name__, self.functionName, self.parameters,
                                          ("\t" + "\t".join(str(self.body).splitlines(True))))

    def __repr__(self):
        self._updateParameters()
        return '{0} {1}({2})'.format(type(self).__name__, self.functionName, self.parameters)


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
        newNode.parent = self
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
    def undefinedSymbols(self):
        result = set()
        definedSymbols = set()
        for a in self.args:
            result.update(a.undefinedSymbols)
            definedSymbols.update(a.symbolsDefined)
        return result - definedSymbols

    def __str__(self):
        return ''.join('{!s}\n'.format(node) for node in self._nodes)

    def __repr__(self):
        return ''.join('{!r}'.format(node) for node in self._nodes)


class PragmaBlock(Block):
    def __init__(self, pragmaLine, listOfNodes):
        super(PragmaBlock, self).__init__(listOfNodes)
        self.pragmaLine = pragmaLine


class LoopOverCoordinate(Node):
    LOOP_COUNTER_NAME_PREFIX = "ctr"

    def __init__(self, body, coordinateToLoopOver, start, stop, step=1):
        self.body = body
        body.parent = self
        self.coordinateToLoopOver = coordinateToLoopOver
        self.start = start
        self.stop = stop
        self.step = step
        self.body.parent = self
        self.prefixLines = []

    def newLoopWithDifferentBody(self, newBody):
        result = LoopOverCoordinate(newBody, self.coordinateToLoopOver, self.start, self.stop, self.step)
        result.prefixLines = [l for l in self.prefixLines]
        return result

    def subs(self, *args, **kwargs):
        self.body.subs(*args, **kwargs)
        if hasattr(self.start, "subs"):
            self.start = self.start.subs(*args, **kwargs)
        if hasattr(self.stop, "subs"):
            self.stop = self.stop.subs(*args, **kwargs)
        if hasattr(self.step, "subs"):
            self.step = self.step.subs(*args, **kwargs)

    @property
    def args(self):
        result = [self.body]
        for e in [self.start, self.stop, self.step]:
            if hasattr(e, "args"):
                result.append(e)
        return result

    @property
    def symbolsDefined(self):
        return set([self.loopCounterSymbol])

    @property
    def undefinedSymbols(self):
        result = self.body.undefinedSymbols
        for possibleSymbol in [self.start, self.stop, self.step]:
            if isinstance(possibleSymbol, Node) or isinstance(possibleSymbol, sp.Basic):
                result.update(possibleSymbol.atoms(sp.Symbol))
        return result - set([self.loopCounterSymbol])

    @staticmethod
    def getLoopCounterName(coordinateToLoopOver):
        return "%s_%s" % (LoopOverCoordinate.LOOP_COUNTER_NAME_PREFIX, coordinateToLoopOver)

    @property
    def loopCounterName(self):
        return LoopOverCoordinate.getLoopCounterName(self.coordinateToLoopOver)

    @staticmethod
    def isLoopCounterSymbol(symbol):
        prefix = LoopOverCoordinate.LOOP_COUNTER_NAME_PREFIX
        if not symbol.name.startswith(prefix):
            return None
        if symbol.dtype != createTypeFromString('int'):
            return None
        coordinate = int(symbol.name[len(prefix)+1:])
        return coordinate

    @staticmethod
    def getLoopCounterSymbol(coordinateToLoopOver):
        return TypedSymbol(LoopOverCoordinate.getLoopCounterName(coordinateToLoopOver), 'int')

    @property
    def loopCounterSymbol(self):
        return LoopOverCoordinate.getLoopCounterSymbol(self.coordinateToLoopOver)

    @property
    def isOutermostLoop(self):
        from pystencils.transformations import getNextParentOfType
        return getNextParentOfType(self, LoopOverCoordinate) is None

    @property
    def isInnermostLoop(self):
        return len(self.atoms(LoopOverCoordinate)) == 0

    def __str__(self):
        return 'loop:{!s} in {!s}:{!s}:{!s}\n{!s}'.format(self.loopCounterName, self.start, self.stop, self.step,
                                                          ("\t" + "\t".join(str(self.body).splitlines(True))))

    def __repr__(self):
        return 'loop:{!s} in {!s}:{!s}:{!s}'.format(self.loopCounterName, self.start, self.stop, self.step)


class SympyAssignment(Node):
    def __init__(self, lhsSymbol, rhsTerm, isConst=True):
        self._lhsSymbol = lhsSymbol
        self.rhs = rhsTerm
        self._isDeclaration = True
        isCast = self._lhsSymbol.func == castFunc
        if isinstance(self._lhsSymbol, Field.Access) or isinstance(self._lhsSymbol, ResolvedFieldAccess) or isCast:
            self._isDeclaration = False
        self._isConst = isConst

    @property
    def lhs(self):
        return self._lhsSymbol

    @lhs.setter
    def lhs(self, newValue):
        self._lhsSymbol = newValue
        self._isDeclaration = True
        isCast = self._lhsSymbol.func == castFunc
        if isinstance(self._lhsSymbol, Field.Access) or isinstance(self._lhsSymbol, sp.Indexed) or isCast:
            self._isDeclaration = False

    def subs(self, *args, **kwargs):
        self.lhs = fastSubs(self.lhs, *args, **kwargs)
        self.rhs = fastSubs(self.rhs, *args, **kwargs)

    @property
    def args(self):
        return [self._lhsSymbol, self.rhs]

    @property
    def symbolsDefined(self):
        if not self._isDeclaration:
            return set()
        return set([self._lhsSymbol])

    @property
    def undefinedSymbols(self):
        result = self.rhs.atoms(sp.Symbol)
        # Add loop counters if there a field accesses
        loopCounters = set()
        for symbol in result:
            if isinstance(symbol, Field.Access):
                for i in range(len(symbol.offsets)):
                    loopCounters.add(LoopOverCoordinate.getLoopCounterSymbol(i))
        result.update(loopCounters)
        result.update(self._lhsSymbol.atoms(sp.Symbol))
        return result

    @property
    def isDeclaration(self):
        return self._isDeclaration

    @property
    def isConst(self):
        return self._isConst

    def replace(self, child, replacement):
        if child == self.lhs:
            replacement.parent = self
            self.lhs = replacement
        elif child == self.rhs:
            replacement.parent = self
            self.rhs = replacement
        else:
            raise ValueError('%s is not in args of %s' % (replacement, self.__class__))

    def __repr__(self):
        return repr(self.lhs) + " = " + repr(self.rhs)


class TemporaryMemoryAllocation(Node):
    def __init__(self, typedSymbol, size):
        self.symbol = typedSymbol
        self.size = size

    @property
    def symbolsDefined(self):
        return set([self.symbol])

    @property
    def undefinedSymbols(self):
        if isinstance(self.size, sp.Basic):
            return self.size.atoms(sp.Symbol)
        else:
            return set()

    @property
    def args(self):
        return [self.symbol]


class TemporaryMemoryFree(Node):
    def __init__(self, typedSymbol):
        self.symbol = typedSymbol

    @property
    def symbolsDefined(self):
        return set()

    @property
    def undefinedSymbols(self):
        return set()

    @property
    def args(self):
        return []


# TODO implement defined & undefinedSymbols
class Conversion(Node):
    def __init__(self, child, dtype, parent=None):
        super(Conversion, self).__init__(parent)
        self._args = [child]
        self.dtype = dtype

    @property
    def args(self):
        """Returns all arguments/children of this node"""
        return self._args

    @args.setter
    def args(self, value):
        self._args = value

    @property
    def symbolsDefined(self):
        """Set of symbols which are defined by this node. """
        return set()

    @property
    def undefinedSymbols(self):
        """Symbols which are use but are not defined inside this node"""
        raise set()

    def __repr__(self):
        return '(%s(%s))' % (repr(self.dtype), repr(self.args[0].dtype)) + repr(self.args)

# TODO Pow


_expr_dict = {'Add': ' + ', 'Mul': ' * ', 'Pow': '**'}


class Expr(Node):
    def __init__(self, args, parent=None):
        super(Expr, self).__init__(parent)
        self._args = list(args)
        self.dtype = None

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        self._args = value

    def replace(self, child, replacements):
        idx = self.args.index(child)
        del self.args[idx]
        if type(replacements) is list:
            for e in replacements:
                e.parent = self
            self.args = self.args[:idx] + replacements + self.args[idx:]
        else:
            replacements.parent = self
            self.args.insert(idx, replacements)

    @property
    def symbolsDefined(self):
        return set()  # Todo fix for symbol analysis

    @property
    def undefinedSymbols(self):
        return set()  # Todo fix for symbol analysis

    def __repr__(self):
        return _expr_dict[self.__class__.__name__].join(repr(arg) for arg in self.args)


class Mul(Expr):
    pass


class Add(Expr):
    pass


class Pow(Expr):
    pass


class Indexed(Expr):
    def __init__(self, args, base, parent=None):
        super(Indexed, self).__init__(args, parent)
        self.base = base
        # Get dtype from label, and unpointer it
        self.dtype = createType(base.label.dtype.baseType)

    def __repr__(self):
        return '%s[%s]' % (self.args[0], self.args[1])


class PointerArithmetic(Expr):
    def __init__(self, args, pointer, parent=None):
        super(PointerArithmetic, self).__init__([args] + [pointer], parent)
        self.pointer = pointer
        self.offset = args
        self.dtype = pointer.dtype

    def __repr__(self):
        return '*(%s + %s)' % (self.pointer, self.args)


class Number(Node, sp.AtomicExpr):
    def __init__(self, number, parent=None):
        super(Number, self).__init__(parent)

        self.dtype, self.value = get_type_from_sympy(number)
        self._args = tuple()

    @property
    def args(self):
        """Returns all arguments/children of this node"""
        return self._args

    @property
    def symbolsDefined(self):
        """Set of symbols which are defined by this node. """
        return set()

    @property
    def undefinedSymbols(self):
        """Symbols which are use but are not defined inside this node"""
        raise set()

    def __repr__(self):
        return repr(self.value)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)


