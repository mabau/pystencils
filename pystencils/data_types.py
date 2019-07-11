import ctypes

import numpy as np
import sympy as sp
from sympy.core.cache import cacheit
from sympy.logic.boolalg import Boolean

from pystencils.cache import memorycache
from pystencils.utils import all_equal

try:
    import llvmlite.ir as ir
except ImportError as e:
    ir = None
    _ir_importerror = e


# noinspection PyPep8Naming
class address_of(sp.Function):
    is_Atom = True

    def __new__(cls, arg):
        obj = sp.Function.__new__(cls, arg)
        return obj

    @property
    def canonical(self):
        if hasattr(self.args[0], 'canonical'):
            return self.args[0].canonical
        else:
            raise NotImplementedError()

    @property
    def is_commutative(self):
        return self.args[0].is_commutative

    @property
    def dtype(self):
        if hasattr(self.args[0], 'dtype'):
            return PointerType(self.args[0].dtype, restrict=True)
        else:
            return PointerType('void', restrict=True)


# noinspection PyPep8Naming
class cast_func(sp.Function):
    is_Atom = True

    def __new__(cls, *args, **kwargs):
        # to work in conditions of sp.Piecewise cast_func has to be of type Boolean as well
        # however, a cast_function should only be a boolean if its argument is a boolean, otherwise this leads
        # to problems when for example comparing cast_func's for equality
        #
        # lhs = bitwise_and(a, cast_func(1, 'int'))
        # rhs = cast_func(0, 'int')
        # print( sp.Ne(lhs, rhs) ) # would give true if all cast_funcs are booleans
        # -> thus a separate class boolean_cast_func is introduced
        if isinstance(args[0], Boolean):
            cls = boolean_cast_func
        return sp.Function.__new__(cls, *args, **kwargs)

    @property
    def canonical(self):
        if hasattr(self.args[0], 'canonical'):
            return self.args[0].canonical
        else:
            raise NotImplementedError()

    @property
    def is_commutative(self):
        return self.args[0].is_commutative

    @property
    def dtype(self):
        return self.args[1]


# noinspection PyPep8Naming
class boolean_cast_func(cast_func, Boolean):
    pass


# noinspection PyPep8Naming
class vector_memory_access(cast_func):
    nargs = (4,)


# noinspection PyPep8Naming
class reinterpret_cast_func(cast_func):
    pass


# noinspection PyPep8Naming
class pointer_arithmetic_func(sp.Function, Boolean):
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
        except (TypeError, ValueError):
            # on error keep the string
            obj._dtype = dtype
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @property
    def dtype(self):
        return self._dtype

    def _hashable_content(self):
        return super()._hashable_content(), hash(self._dtype)

    def __getnewargs__(self):
        return self.name, self.dtype


def create_type(specification):
    """Creates a subclass of Type according to a string or an object of subclass Type.

    Args:
        specification: Type object, or a string

    Returns:
        Type object, or a new Type object parsed from the string
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
    """Creates a new Type object from a c-like string specification.

    Args:
        specification: Specification string

    Returns:
        Type object
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

    if any(t.is_float() for t in types):
        types = tuple(t for t in types if t.is_float())
    # use numpy collation -> create type from numpy type -> and, put vector type around if necessary
    result_numpy_type = np.result_type(*(t.numpy_dtype for t in types))
    result = BasicType(result_numpy_type)
    if vector_type:
        result = VectorType(result, vector_type[0].width)
    return result


@memorycache(maxsize=2048)
def get_type_of_expression(expr):
    from pystencils.astnodes import ResolvedFieldAccess
    from pystencils.cpu.vectorization import vec_all, vec_any

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
        raise ValueError("All symbols inside this expression have to be typed! ", str(expr))
    elif isinstance(expr, cast_func):
        return expr.args[1]
    elif isinstance(expr, vec_any) or isinstance(expr, vec_all):
        return create_type("bool")
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
    elif isinstance(expr, sp.Pow):
        return get_type_of_expression(expr.args[0])
    elif isinstance(expr, sp.Expr):
        types = tuple(get_type_of_expression(a) for a in expr.args)
        return collate_types(types)

    raise NotImplementedError("Could not determine type for", expr, type(expr))


class Type(sp.Basic):
    is_Atom = True

    def __new__(cls, *args, **kwargs):
        return sp.Basic.__new__(cls)

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
            raise NotImplementedError("Can map numpy to C name for %s" % (name,))

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
    instruction_set = None

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
        if self.instruction_set is None:
            return "%s[%d]" % (self.base_type, self.width)
        else:
            if self.base_type == create_type("int64"):
                return self.instruction_set['int']
            elif self.base_type == create_type("float64"):
                return self.instruction_set['double']
            elif self.base_type == create_type("float32"):
                return self.instruction_set['float']
            elif self.base_type == create_type("bool"):
                return self.instruction_set['bool']
            else:
                raise NotImplementedError()

    def __hash__(self):
        return hash((self.base_type, self.width))

    def __getnewargs__(self):
        return self._base_type, self.width


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
        components = [str(self.base_type), '*']
        if self.restrict:
            components.append('RESTRICT')
        if self.const:
            components.append("const")
        return " ".join(components)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self._base_type, self.const, self.restrict))


class StructType:
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
