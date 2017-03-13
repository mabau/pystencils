import ctypes
import sympy as sp
import numpy as np
from sympy.core.cache import cacheit


class TypedSymbol(sp.Symbol):

    def __new__(cls, name, *args, **kwds):
        obj = TypedSymbol.__xnew_cached_(cls, name, *args, **kwds)
        return obj

    def __new_stage2__(cls, name, dtype):
        obj = super(TypedSymbol, cls).__xnew__(cls, name)
        obj._dtype = createType(dtype)
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @property
    def dtype(self):
        return self._dtype

    def _hashable_content(self):
        superClassContents = list(super(TypedSymbol, self)._hashable_content())
        t = tuple(superClassContents + [hash(repr(self._dtype))])
        return t

    def __getnewargs__(self):
        return self.name, self.dtype


def createType(specification):
    if isinstance(specification, Type):
        return specification
    elif isinstance(specification, str):
        return createTypeFromString(specification)
    else:
        npDataType = np.dtype(specification)
        return BasicType(npDataType, const=False)


def createTypeFromString(specification):
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
    baseType = BasicType(basePart[0], const)

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


def getBaseType(type):
    while type.baseType is not None:
        type = type.baseType
    return type


def toCtypes(dataType):
    if isinstance(dataType, PointerType):
        return ctypes.POINTER(toCtypes(dataType.baseType))
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


class Type(object):
    pass


class BasicType(Type):
    @staticmethod
    def numpyNameToC(name):
        if name == 'float64': return 'double'
        elif name == 'float32': return 'float'
        elif name.startswith('int'):
            width = int(name[len("int"):])
            return "int%d_t" % (width,)
        elif name.startswith('uint'):
            width = int(name[len("int"):])
            return "uint%d_t" % (width,)
        elif name == 'bool':
            return 'bool'
        else:
            raise NotImplemented("Can map numpy to C name for %s" % (name,))

    def __init__(self, dtype, const=False):
        self.const = const
        self._dtype = np.dtype(dtype)
        assert self._dtype.fields is None, "Tried to initialize NativeType with a structured type"
        assert self._dtype.hasobject is False
        assert self._dtype.subdtype is None

    @property
    def baseType(self):
        return None

    @property
    def numpyDtype(self):
        return self._dtype

    def __str__(self):
        result = BasicType.numpyNameToC(str(self._dtype))
        if self.const:
            result += " const"
        return result

    def __eq__(self, other):
        if not isinstance(other, BasicType):
            return False
        else:
            return (self.numpyDtype, self.const) == (other.numpyDtype, other.const)

    def __hash__(self):
        return hash(str(self))


class PointerType(Type):
    def __init__(self, baseType, const=False, restrict=True):
        self._baseType = baseType
        self.const = const
        self.restrict = restrict

    @property
    def alias(self):
        return not self.restrict

    @property
    def baseType(self):
        return self._baseType

    def __eq__(self, other):
        if not isinstance(other, PointerType):
            return False
        else:
            return (self.baseType, self.const, self.restrict) == (other.baseType, other.const, other.restrict)

    def __str__(self):
        return "%s * %s%s" % (self.baseType, "RESTRICT " if self.restrict else "", "const " if self.const else "")

    def __hash__(self):
        return hash(str(self))


class StructType(object):
    def __init__(self, numpyType):
        self._dtype = np.dtype(numpyType)

