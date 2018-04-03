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
from pystencils.utils import all_equal


# to work in conditions of sp.Piecewise castFunc has to be of type Relational as well
class cast_func(sp.Function, sp.Rel):
    @property
    def canonical(self):
        if hasattr(self.args[0], 'canonical'):
            return self.args[0].canonical
        else:
            raise NotImplementedError()

    @property
    def is_commutative(self):
        return self.args[0].is_commutative


class pointer_arithmetic_func(sp.Function, sp.Rel):

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
            obj._dtype = create_type(dtype)
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
        super_class_contents = list(super(TypedSymbol, self)._hashable_content())
        return tuple(super_class_contents + [hash(self._dtype)])

    def __getnewargs__(self):
        return self.name, self.dtype


def create_type(specification):
    """
    Create a subclass of Type according to a string or an object of subclass Type
    :param specification: Type object, or a string
    :return: Type object, or a new Type object parsed from the string
    """
    if isinstance(specification, Type):
        return specification
    else:
        numpy_dtype = np.dtype(specification)
        if numpy_dtype.fields is None:
            return BasicType(numpy_dtype, const=False)
        else:
            return StructType(numpy_dtype, const=False)


@memorycache(maxsize=64)
def create_composite_type_from_string(specification):
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
    base_part = parts.pop(0)
    const = False
    if 'const' in base_part:
        const = True
        base_part.remove('const')
    assert len(base_part) == 1
    if base_part[0][-1] == "*":
        base_part[0] = base_part[0][:-1]
        parts.append('*')
    current_type = BasicType(np.dtype(base_part[0]), const)
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
        current_type = PointerType(current_type, const, restrict)
    return current_type


def get_base_type(data_type):
    while data_type.base_type is not None:
        data_type = data_type.base_type
    return data_type


def to_ctypes(data_type):
    """
    Transforms a given Type into ctypes
    :param data_type: Subclass of Type
    :return: ctypes type object
    """
    if isinstance(data_type, PointerType):
        return ctypes.POINTER(to_ctypes(data_type.base_type))
    elif isinstance(data_type, StructType):
        return ctypes.POINTER(ctypes.c_uint8)
    else:
        return to_ctypes.map[data_type.numpy_dtype]


to_ctypes.map = {
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
        return to_llvm_type(data_type.base_type).as_pointer()
    else:
        return to_llvm_type.map[data_type.numpy_dtype]


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


def peel_off_type(dtype, type_to_peel_off):
    while type(dtype) is type_to_peel_off:
        dtype = dtype.base_type
    return dtype


def collate_types(types):
    """
    Takes a sequence of types and returns their "common type" e.g. (float, double, float) -> double
    Uses the collation rules from numpy.
    """

    # Pointer arithmetic case i.e. pointer + integer is allowed
    if any(type(t) is PointerType for t in types):
        pointer_type = None
        for t in types:
            if type(t) is PointerType:
                if pointer_type is not None:
                    raise ValueError("Cannot collate the combination of two pointer types")
                pointer_type = t
            elif type(t) is BasicType:
                if not (t.is_int() or t.is_uint()):
                    raise ValueError("Invalid pointer arithmetic")
            else:
                raise ValueError("Invalid pointer arithmetic")
        return pointer_type

    # peel of vector types, if at least one vector type occurred the result will also be the vector type
    vector_type = [t for t in types if type(t) is VectorType]
    if not all_equal(t.width for t in vector_type):
        raise ValueError("Collation failed because of vector types with different width")
    types = [peel_off_type(t, VectorType) for t in types]

    # now we should have a list of basic types - struct types are not yet supported
    assert all(type(t) is BasicType for t in types)

    # use numpy collation -> create type from numpy type -> and, put vector type around if necessary
    result_numpy_type = np.result_type(*(t.numpy_dtype for t in types))
    result = BasicType(result_numpy_type)
    if vector_type:
        result = VectorType(result, vector_type[0].width)
    return result


@memorycache(maxsize=2048)
def get_type_of_expression(expr):
    from pystencils.astnodes import ResolvedFieldAccess
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Integer):
        return create_type("int")
    elif isinstance(expr, sp.Rational) or isinstance(expr, sp.Float):
        return create_type("double")
    elif isinstance(expr, ResolvedFieldAccess):
        return expr.field.dtype
    elif isinstance(expr, TypedSymbol):
        return expr.dtype
    elif isinstance(expr, sp.Symbol):
        raise ValueError("All symbols inside this expression have to be typed!")
    elif hasattr(expr, 'func') and expr.func == cast_func:
        return expr.args[1]
    elif hasattr(expr, 'func') and expr.func == sp.Piecewise:
        collated_result_type = collate_types(tuple(get_type_of_expression(a[0]) for a in expr.args))
        collated_condition_type = collate_types(tuple(get_type_of_expression(a[1]) for a in expr.args))
        if type(collated_condition_type) is VectorType and type(collated_result_type) is not VectorType:
            collated_result_type = VectorType(collated_result_type, width=collated_condition_type.width)
        return collated_result_type
    elif isinstance(expr, sp.Indexed):
        typed_symbol = expr.base.label
        return typed_symbol.dtype.base_type
    elif isinstance(expr, sp.boolalg.Boolean) or isinstance(expr, sp.boolalg.BooleanFunction):
        # if any arg is of vector type return a vector boolean, else return a normal scalar boolean
        result = create_type("bool")
        vec_args = [get_type_of_expression(a) for a in expr.args if isinstance(get_type_of_expression(a), VectorType)]
        if vec_args:
            result = VectorType(result, width=vec_args[0].width)
        return result
    elif isinstance(expr, sp.Expr):
        types = tuple(get_type_of_expression(a) for a in expr.args)
        return collate_types(types)

    raise NotImplementedError("Could not determine type for", expr, type(expr))


class Type(sp.Basic):
    def __new__(cls, *args, **kwargs):
        return sp.Basic.__new__(cls)

    def __lt__(self, other):  # deprecated
        # Needed for sorting the types inside an expression
        if isinstance(self, BasicType):
            if isinstance(other, BasicType):
                return self.numpy_dtype > other.numpy_dtype  # TODO const
            elif isinstance(other, PointerType):
                return False
            else:  # isinstance(other, StructType):
                raise NotImplementedError("Struct type comparison is not yet implemented")
        elif isinstance(self, PointerType):
            if isinstance(other, BasicType):
                return True
            elif isinstance(other, PointerType):
                return self.base_type > other.base_type  # TODO const, restrict
            else:  # isinstance(other, StructType):
                raise NotImplementedError("Struct type comparison is not yet implemented")
        elif isinstance(self, StructType):
            raise NotImplementedError("Struct type comparison is not yet implemented")
        else:
            raise NotImplementedError

    def _sympystr(self, *args, **kwargs):
        return str(self)


class BasicType(Type):
    @staticmethod
    def numpy_name_to_c(name):
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
            self._dtype = dtype.numpy_dtype
        else:
            self._dtype = np.dtype(dtype)
        assert self._dtype.fields is None, "Tried to initialize NativeType with a structured type"
        assert self._dtype.hasobject is False
        assert self._dtype.subdtype is None

    def __getnewargs__(self):
        return self.numpy_dtype, self.const

    @property
    def base_type(self):
        return None

    @property
    def numpy_dtype(self):
        return self._dtype

    @property
    def item_size(self):
        return 1

    def is_int(self):
        return self.numpy_dtype in np.sctypes['int']

    def is_float(self):
        return self.numpy_dtype in np.sctypes['float']

    def is_uint(self):
        return self.numpy_dtype in np.sctypes['uint']

    def is_complex(self):
        return self.numpy_dtype in np.sctypes['complex']

    def is_other(self):
        return self.numpy_dtype in np.sctypes['others']

    @property
    def base_name(self):
        return BasicType.numpy_name_to_c(str(self._dtype))

    def __str__(self):
        result = BasicType.numpy_name_to_c(str(self._dtype))
        if self.const:
            result += " const"
        return result

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, BasicType):
            return False
        else:
            return (self.numpy_dtype, self.const) == (other.numpy_dtype, other.const)

    def __hash__(self):
        return hash(str(self))


class VectorType(Type):
    instructionSet = None

    def __init__(self, base_type, width=4):
        self._base_type = base_type
        self.width = width

    @property
    def base_type(self):
        return self._base_type

    @property
    def item_size(self):
        return self.width * self.base_type.item_size

    def __eq__(self, other):
        if not isinstance(other, VectorType):
            return False
        else:
            return (self.base_type, self.width) == (other.base_type, other.width)

    def __str__(self):
        if self.instructionSet is None:
            return "%s[%d]" % (self.base_type, self.width)
        else:
            if self.base_type == create_type("int64"):
                return self.instructionSet['int']
            elif self.base_type == create_type("float64"):
                return self.instructionSet['double']
            elif self.base_type == create_type("float32"):
                return self.instructionSet['float']
            elif self.base_type == create_type("bool"):
                return self.instructionSet['bool']
            else:
                raise NotImplementedError()

    def __hash__(self):
        return hash((self.base_type, self.width))


class PointerType(Type):
    def __init__(self, base_type, const=False, restrict=True):
        self._base_type = base_type
        self.const = const
        self.restrict = restrict

    def __getnewargs__(self):
        return self.base_type, self.const, self.restrict

    @property
    def alias(self):
        return not self.restrict

    @property
    def base_type(self):
        return self._base_type

    @property
    def item_size(self):
        return self.base_type.item_size

    def __eq__(self, other):
        if not isinstance(other, PointerType):
            return False
        else:
            return (self.base_type, self.const, self.restrict) == (other.base_type, other.const, other.restrict)

    def __str__(self):
        return "%s *%s%s" % (self.base_type, " RESTRICT " if self.restrict else "", " const " if self.const else "")

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self._base_type, self.const, self.restrict))


class StructType(object):
    def __init__(self, numpy_type, const=False):
        self.const = const
        self._dtype = np.dtype(numpy_type)

    def __getnewargs__(self):
        return self.numpy_dtype, self.const

    @property
    def base_type(self):
        return None

    @property
    def numpy_dtype(self):
        return self._dtype

    @property
    def item_size(self):
        return self.numpy_dtype.itemsize

    def get_element_offset(self, element_name):
        return self.numpy_dtype.fields[element_name][1]

    def get_element_type(self, element_name):
        np_element_type = self.numpy_dtype.fields[element_name][0]
        return BasicType(np_element_type, self.const)

    def has_element(self, element_name):
        return element_name in self.numpy_dtype.fields

    def __eq__(self, other):
        if not isinstance(other, StructType):
            return False
        else:
            return (self.numpy_dtype, self.const) == (other.numpy_dtype, other.const)

    def __str__(self):
        # structs are handled byte-wise
        result = "uint8_t"
        if self.const:
            result += " const"
        return result

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.numpy_dtype, self.const))

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
        return create_type('double'), float(node)
    elif isinstance(node, sp.Integer):
        return create_type('int'), int(node)
    elif isinstance(node, sp.Rational):
        # TODO is it always float?
        return create_type('double'), float(node.p / node.q)
    else:
        raise TypeError(node, ' is not a supported type (yet)!')
