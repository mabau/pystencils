import ctypes
import sympy as sp
import numpy as np
try:
    import llvmlite.ir as ir
except ImportError as e:
    ir = None
    _ir_importerror = e
from sympy.core.cache import cacheit

from pystencils.cache import memorycache
from pystencils.utils import allEqual


# to work in conditions of sp.Piecewise castFunc has to be of type Relational as well
class castFunc(sp.Function, sp.Rel):
    @property
    def canonical(self):
        if hasattr(self.args[0], 'canonical'):
            return self.args[0].canonical
        else:
            raise NotImplementedError()

    @property
    def is_commutative(self):
        return self.args[0].is_commutative


class pointerArithmeticFunc(sp.Function, sp.Rel):

    @property
    def canonical(self):
        if hasattr(self.args[0], 'canonical'):
            return self.args[0].canonical
        else:
            raise NotImplementedError()


class TypedSymbol(sp.Symbol):
    def __new__(cls, *args, **kwds):
        obj = TypedSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, name, dtype):
        obj = super(TypedSymbol, cls).__xnew__(cls, name)
        try:
            obj._dtype = createType(dtype)
        except TypeError:
            # on error keep the string
            obj._dtype = dtype
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @property
    def dtype(self):
        return self._dtype

    def _hashable_content(self):
        superClassContents = list(super(TypedSymbol, self)._hashable_content())
        return tuple(superClassContents + [hash(str(self._dtype))])

    def __getnewargs__(self):
        return self.name, self.dtype


def createType(specification):
    """
    Create a subclass of Type according to a string or an object of subclass Type
    :param specification: Type object, or a string
    :return: Type object, or a new Type object parsed from the string
    """
    if isinstance(specification, Type):
        return specification
    elif isinstance(specification, str):
        return createTypeFromString(specification)
    else:
        npDataType = np.dtype(specification)
        if npDataType.fields is None:
            return BasicType(npDataType, const=False)
        else:
            return StructType(npDataType, const=False)


@memorycache(maxsize=64)
def createTypeFromString(specification):
    """
    Creates a new Type object from a c-like string specification
    :param specification: Specification string
    :return: Type object
    """
    specification = specification.lower().split()
    parts = []
    current = []
    for s in specification:
        if s == '*':
            parts.append(current)
            current = [s]
        else:
            current.append(s)
    if len(current) > 0:
        parts.append(current)
        # Parse native part
    basePart = parts.pop(0)
    const = False
    if 'const' in basePart:
        const = True
        basePart.remove('const')
    assert len(basePart) == 1
    if basePart[0][-1] == "*":
        basePart[0] = basePart[0][:-1]
        parts.append('*')
    try:
        baseType = BasicType(basePart[0], const)
    except TypeError:
        baseType = BasicType(createTypeFromString.map[basePart[0]], const)
    currentType = baseType
    # Parse pointer parts
    for part in parts:
        restrict = False
        const = False
        if 'restrict' in part:
            restrict = True
            part.remove('restrict')
        if 'const' in part:
            const = True
            part.remove("const")
        assert len(part) == 1 and part[0] == '*'
        currentType = PointerType(currentType, const, restrict)
    return currentType

createTypeFromString.map = {
    'i64': np.int64,
    'i32': np.int32,
    'i16': np.int16,
    'i8': np.int8,
}


def getBaseType(type):
    while type.baseType is not None:
        type = type.baseType
    return type


def toCtypes(dataType):
    """
    Transforms a given Type into ctypes
    :param dataType: Subclass of Type
    :return: ctypes type object
    """
    if isinstance(dataType, PointerType):
        return ctypes.POINTER(toCtypes(dataType.baseType))
    elif isinstance(dataType, StructType):
        return ctypes.POINTER(ctypes.c_uint8)
    else:
        return toCtypes.map[dataType.numpyDtype]

toCtypes.map = {
    np.dtype(np.int8): ctypes.c_int8,
    np.dtype(np.int16): ctypes.c_int16,
    np.dtype(np.int32): ctypes.c_int32,
    np.dtype(np.int64): ctypes.c_int64,

    np.dtype(np.uint8): ctypes.c_uint8,
    np.dtype(np.uint16): ctypes.c_uint16,
    np.dtype(np.uint32): ctypes.c_uint32,
    np.dtype(np.uint64): ctypes.c_uint64,

    np.dtype(np.float32): ctypes.c_float,
    np.dtype(np.float64): ctypes.c_double,
}


def ctypes_from_llvm(data_type):
    if not ir:
        raise _ir_importerror
    if isinstance(data_type, ir.PointerType):
        ctype = ctypes_from_llvm(data_type.pointee)
        if ctype is None:
            return ctypes.c_void_p
        else:
            return ctypes.POINTER(ctype)
    elif isinstance(data_type, ir.IntType):
        if data_type.width == 8:
            return ctypes.c_int8
        elif data_type.width == 16:
            return ctypes.c_int16
        elif data_type.width == 32:
            return ctypes.c_int32
        elif data_type.width == 64:
            return ctypes.c_int64
        else:
            raise ValueError("Int width %d is not supported" % data_type.width)
    elif isinstance(data_type, ir.FloatType):
        return ctypes.c_float
    elif isinstance(data_type, ir.DoubleType):
        return ctypes.c_double
    elif isinstance(data_type, ir.VoidType):
        return None  # Void type is not supported by ctypes
    else:
        raise NotImplementedError('Data type %s of %s is not supported yet' % (type(data_type), data_type))


def to_llvm_type(data_type):
    """
    Transforms a given type into ctypes
    :param data_type: Subclass of Type
    :return: llvmlite type object
    """
    if not ir:
        raise _ir_importerror
    if isinstance(data_type, PointerType):
        return to_llvm_type(data_type.baseType).as_pointer()
    else:
        return to_llvm_type.map[data_type.numpyDtype]

if ir:
    to_llvm_type.map = {
        np.dtype(np.int8): ir.IntType(8),
        np.dtype(np.int16): ir.IntType(16),
        np.dtype(np.int32): ir.IntType(32),
        np.dtype(np.int64): ir.IntType(64),

        np.dtype(np.uint8): ir.IntType(8),
        np.dtype(np.uint16): ir.IntType(16),
        np.dtype(np.uint32): ir.IntType(32),
        np.dtype(np.uint64): ir.IntType(64),

        np.dtype(np.float32): ir.FloatType(),
        np.dtype(np.float64): ir.DoubleType(),
    }

def peelOffType(dtype, typeToPeelOff):
    while type(dtype) is typeToPeelOff:
        dtype = dtype.baseType
    return dtype


def collateTypes(types):
    """
    Takes a sequence of types and returns their "common type" e.g. (float, double, float) -> double
    Uses the collation rules from numpy.
    """

    # Pointer arithmetic case i.e. pointer + integer is allowed
    if any(type(t) is PointerType for t in types):
        pointerType = None
        for t in types:
            if type(t) is PointerType:
                if pointerType is not None:
                    raise ValueError("Cannot collate the combination of two pointer types")
                pointerType = t
            elif type(t) is BasicType:
                if not (t.is_int() or t.is_uint()):
                    raise ValueError("Invalid pointer arithmetic")
            else:
                raise ValueError("Invalid pointer arithmetic")
        return pointerType

    # peel of vector types, if at least one vector type occurred the result will also be the vector type
    vectorType = [t for t in types if type(t) is VectorType]
    if not allEqual(t.width for t in vectorType):
        raise ValueError("Collation failed because of vector types with different width")
    types = [peelOffType(t, VectorType) for t in types]

    # now we should have a list of basic types - struct types are not yet supported
    assert all(type(t) is BasicType for t in types)

    # use numpy collation -> create type from numpy type -> and, put vector type around if necessary
    resultNumpyType = np.result_type(*(t.numpyDtype for t in types))
    result = BasicType(resultNumpyType)
    if vectorType:
        result = VectorType(result, vectorType[0].width)
    return result


@memorycache(maxsize=2048)
def getTypeOfExpression(expr):
    from pystencils.astnodes import ResolvedFieldAccess
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Integer):
        return createTypeFromString("int")
    elif isinstance(expr, sp.Rational) or isinstance(expr, sp.Float):
        return createTypeFromString("double")
    elif isinstance(expr, ResolvedFieldAccess):
        return expr.field.dtype
    elif isinstance(expr, TypedSymbol):
        return expr.dtype
    elif isinstance(expr, sp.Symbol):
        raise ValueError("All symbols inside this expression have to be typed!")
    elif hasattr(expr, 'func') and expr.func == castFunc:
        return expr.args[1]
    elif hasattr(expr, 'func') and expr.func == sp.Piecewise:
        collatedResultType = collateTypes(tuple(getTypeOfExpression(a[0]) for a in expr.args))
        collatedConditionType = collateTypes(tuple(getTypeOfExpression(a[1]) for a in expr.args))
        if type(collatedConditionType) is VectorType and type(collatedResultType) is not VectorType:
            collatedResultType = VectorType(collatedResultType, width=collatedConditionType.width)
        return collatedResultType
    elif isinstance(expr, sp.Indexed):
        typedSymbol = expr.base.label
        return typedSymbol.dtype.baseType
    elif isinstance(expr, sp.boolalg.Boolean) or isinstance(expr, sp.boolalg.BooleanFunction):
        # if any arg is of vector type return a vector boolean, else return a normal scalar boolean
        result = createTypeFromString("bool")
        vecArgs = [getTypeOfExpression(a) for a in expr.args if isinstance(getTypeOfExpression(a), VectorType)]
        if vecArgs:
            result = VectorType(result, width=vecArgs[0].width)
        return result
    elif isinstance(expr, sp.Expr):
        types = tuple(getTypeOfExpression(a) for a in expr.args)
        return collateTypes(types)

    raise NotImplementedError("Could not determine type for", expr, type(expr))


class Type(sp.Basic):
    def __new__(cls, *args, **kwargs):
        return sp.Basic.__new__(cls)

    def __lt__(self, other):  # deprecated
        # Needed for sorting the types inside an expression
        if isinstance(self, BasicType):
            if isinstance(other, BasicType):
                return self.numpyDtype > other.numpyDtype  # TODO const
            elif isinstance(other, PointerType):
                return False
            else:  # isinstance(other, StructType):
                raise NotImplementedError("Struct type comparison is not yet implemented")
        elif isinstance(self, PointerType):
            if isinstance(other, BasicType):
                return True
            elif isinstance(other, PointerType):
                return self.baseType > other.baseType  # TODO const, restrict
            else:  # isinstance(other, StructType):
                raise NotImplementedError("Struct type comparison is not yet implemented")
        elif isinstance(self, StructType):
            raise NotImplementedError("Struct type comparison is not yet implemented")
        else:
            raise NotImplementedError

    def _sympystr(self, *args, **kwargs):
        return str(self)

    def _sympystr(self, *args, **kwargs):
        return str(self)


class BasicType(Type):
    @staticmethod
    def numpyNameToC(name):
        if name == 'float64':
            return 'double'
        elif name == 'float32':
            return 'float'
        elif name.startswith('int'):
            width = int(name[len("int"):])
            return "int%d_t" % (width,)
        elif name.startswith('uint'):
            width = int(name[len("uint"):])
            return "uint%d_t" % (width,)
        elif name == 'bool':
            return 'bool'
        else:
            raise NotImplemented("Can map numpy to C name for %s" % (name,))

    def __init__(self, dtype, const=False):
        self.const = const
        if isinstance(dtype, Type):
            self._dtype = dtype.numpyDtype
        else:
            self._dtype = np.dtype(dtype)
        assert self._dtype.fields is None, "Tried to initialize NativeType with a structured type"
        assert self._dtype.hasobject is False
        assert self._dtype.subdtype is None

    def __getnewargs__(self):
        return self.numpyDtype, self.const

    @property
    def baseType(self):
        return None

    @property
    def numpyDtype(self):
        return self._dtype

    @property
    def itemSize(self):
        return 1

    def is_int(self):
        return self.numpyDtype in np.sctypes['int']

    def is_float(self):
        return self.numpyDtype in np.sctypes['float']

    def is_uint(self):
        return self.numpyDtype in np.sctypes['uint']

    def is_comlex(self):
        return self.numpyDtype in np.sctypes['complex']

    def is_other(self):
        return self.numpyDtype in np.sctypes['others']

    @property
    def baseName(self):
        return BasicType.numpyNameToC(str(self._dtype))

    def __str__(self):
        result = BasicType.numpyNameToC(str(self._dtype))
        if self.const:
            result += " const"
        return result

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, BasicType):
            return False
        else:
            return (self.numpyDtype, self.const) == (other.numpyDtype, other.const)

    def __hash__(self):
        return hash(str(self))


class VectorType(Type):
    instructionSet = None

    def __init__(self, baseType, width=4):
        self._baseType = baseType
        self.width = width

    @property
    def baseType(self):
        return self._baseType

    @property
    def itemSize(self):
        return self.width * self.baseType.itemSize

    def __eq__(self, other):
        if not isinstance(other, VectorType):
            return False
        else:
            return (self.baseType, self.width) == (other.baseType, other.width)

    def __str__(self):
        if self.instructionSet is None:
            return "%s[%d]" % (self.baseType, self.width)
        else:
            if self.baseType == createTypeFromString("int64"):
                return self.instructionSet['int']
            elif self.baseType == createTypeFromString("double"):
                return self.instructionSet['double']
            elif self.baseType == createTypeFromString("float"):
                return self.instructionSet['float']
            elif self.baseType == createTypeFromString("bool"):
                return self.instructionSet['bool']
            else:
                raise NotImplementedError()

    def __hash__(self):
        return hash(str(self))


class PointerType(Type):
    def __init__(self, baseType, const=False, restrict=True):
        self._baseType = baseType
        self.const = const
        self.restrict = restrict

    def __getnewargs__(self):
        return self.baseType, self.const, self.restrict

    @property
    def alias(self):
        return not self.restrict

    @property
    def baseType(self):
        return self._baseType

    @property
    def itemSize(self):
        return self.baseType.itemSize

    def __eq__(self, other):
        if not isinstance(other, PointerType):
            return False
        else:
            return (self.baseType, self.const, self.restrict) == (other.baseType, other.const, other.restrict)

    def __str__(self):
        return "%s *%s%s" % (self.baseType, " RESTRICT " if self.restrict else "", " const " if self.const else "")

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))


class StructType(object):
    def __init__(self, numpyType, const=False):
        self.const = const
        self._dtype = np.dtype(numpyType)

    def __getnewargs__(self):
        return self.numpyDtype, self.const

    @property
    def baseType(self):
        return None

    @property
    def numpyDtype(self):
        return self._dtype

    @property
    def itemSize(self):
        return self.numpyDtype.itemsize

    def getElementOffset(self, elementName):
        return self.numpyDtype.fields[elementName][1]

    def getElementType(self, elementName):
        npElementType = self.numpyDtype.fields[elementName][0]
        return BasicType(npElementType, self.const)

    def hasElement(self, elementName):
        return elementName in self.numpyDtype.fields

    def __eq__(self, other):
        if not isinstance(other, StructType):
            return False
        else:
            return (self.numpyDtype, self.const) == (other.numpyDtype, other.const)

    def __str__(self):
        # structs are handled byte-wise
        result = "uint8_t"
        if self.const:
            result += " const"
        return result

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.numpyDtype, self.const))

    # TODO this should not work at all!!!
    def __gt__(self, other):
        if self.ptr and not other.ptr:
            return True
        if self.dtype > other.dtype:
            return True


def get_type_from_sympy(node):
    """
    Creates a Type object from a Sympy object
    :param node: Sympy object
    :return: Type object
    """
    # Rational, NumberSymbol?
    # Zero, One, NegativeOne )= Integer
    # Half )= Rational
    # NAN, Infinity, Negative Inifinity,
    # Exp1, Imaginary Unit, Pi, EulerGamma, Catalan, Golden Ratio
    # Pow, Mul, Add, Mod, Relational
    if not isinstance(node, sp.Number):
        raise TypeError(node, 'is not a sp.Number')

    if isinstance(node, sp.Float) or isinstance(node, sp.RealNumber):
        return createType('double'), float(node)
    elif isinstance(node, sp.Integer):
        return createType('int'), int(node)
    elif isinstance(node, sp.Rational):
        # TODO is it always float?
        return createType('double'), float(node.p/node.q)
    else:
        raise TypeError(node, ' is not a supported type (yet)!')
