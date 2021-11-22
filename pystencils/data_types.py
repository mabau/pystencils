import ctypes
from collections import defaultdict
from functools import partial
from typing import Tuple

import numpy as np
import sympy as sp
import sympy.codegen.ast
from sympy.core.cache import cacheit
from sympy.logic.boolalg import Boolean, BooleanFunction

import pystencils
from pystencils.cache import memorycache, memorycache_if_hashable
from pystencils.utils import all_equal


def typed_symbols(names, dtype, *args):
    symbols = sp.symbols(names, *args)
    if isinstance(symbols, Tuple):
        return tuple(TypedSymbol(str(s), dtype) for s in symbols)
    else:
        return TypedSymbol(str(symbols), dtype)


def type_all_numbers(expr, dtype):
    substitutions = {a: cast_func(a, dtype) for a in expr.atoms(sp.Number)}
    return expr.subs(substitutions)


def matrix_symbols(names, dtype, rows, cols):
    if isinstance(names, str):
        names = names.replace(' ', '').split(',')

    matrices = []
    for n in names:
        symbols = typed_symbols(f"{n}:{rows * cols}", dtype)
        matrices.append(sp.Matrix(rows, cols, lambda i, j: symbols[i * cols + j]))

    return tuple(matrices)


def assumptions_from_dtype(dtype):
    """Derives SymPy assumptions from :class:`BasicType` or a Numpy dtype

    Args:
        dtype (BasicType, np.dtype): a Numpy data type
    Returns:
        A dict of SymPy assumptions
    """
    if hasattr(dtype, 'numpy_dtype'):
        dtype = dtype.numpy_dtype

    assumptions = dict()

    try:
        if np.issubdtype(dtype, np.integer):
            assumptions.update({'integer': True})

        if np.issubdtype(dtype, np.unsignedinteger):
            assumptions.update({'negative': False})

        if np.issubdtype(dtype, np.integer) or \
                np.issubdtype(dtype, np.floating):
            assumptions.update({'real': True})
    except Exception:
        pass

    return assumptions


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
        if len(args) != 2:
            pass
        expr, dtype, *other_args = args
        if not isinstance(dtype, Type):
            dtype = create_type(dtype)
        # to work in conditions of sp.Piecewise cast_func has to be of type Boolean as well
        # however, a cast_function should only be a boolean if its argument is a boolean, otherwise this leads
        # to problems when for example comparing cast_func's for equality
        #
        # lhs = bitwise_and(a, cast_func(1, 'int'))
        # rhs = cast_func(0, 'int')
        # print( sp.Ne(lhs, rhs) ) # would give true if all cast_funcs are booleans
        # -> thus a separate class boolean_cast_func is introduced
        if isinstance(expr, Boolean) and (not isinstance(expr, TypedSymbol) or expr.dtype == BasicType(bool)):
            cls = boolean_cast_func

        return sp.Function.__new__(cls, expr, dtype, *other_args, **kwargs)

    @property
    def canonical(self):
        if hasattr(self.args[0], 'canonical'):
            return self.args[0].canonical
        else:
            raise NotImplementedError()

    @property
    def is_commutative(self):
        return self.args[0].is_commutative

    def _eval_evalf(self, *args, **kwargs):
        return self.args[0].evalf()

    @property
    def dtype(self):
        return self.args[1]

    @property
    def is_integer(self):
        """
        Uses Numpy type hierarchy to determine :func:`sympy.Expr.is_integer` predicate

        For reference: Numpy type hierarchy https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.scalars.html
        """
        if hasattr(self.dtype, 'numpy_dtype'):
            return np.issubdtype(self.dtype.numpy_dtype, np.integer) or super().is_integer
        else:
            return super().is_integer

    @property
    def is_negative(self):
        """
        See :func:`.TypedSymbol.is_integer`
        """
        if hasattr(self.dtype, 'numpy_dtype'):
            if np.issubdtype(self.dtype.numpy_dtype, np.unsignedinteger):
                return False

        return super().is_negative

    @property
    def is_nonnegative(self):
        """
        See :func:`.TypedSymbol.is_integer`
        """
        if self.is_negative is False:
            return True
        else:
            return super().is_nonnegative

    @property
    def is_real(self):
        """
        See :func:`.TypedSymbol.is_integer`
        """
        if hasattr(self.dtype, 'numpy_dtype'):
            return np.issubdtype(self.dtype.numpy_dtype, np.integer) or \
                np.issubdtype(self.dtype.numpy_dtype, np.floating) or \
                super().is_real
        else:
            return super().is_real


# noinspection PyPep8Naming
class boolean_cast_func(cast_func, Boolean):
    pass


# noinspection PyPep8Naming
class vector_memory_access(cast_func):
    # Arguments are: read/write expression, type, aligned, nontemporal, mask (or none), stride
    nargs = (6,)


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

    def __new_stage2__(cls, name, dtype, **kwargs):
        assumptions = assumptions_from_dtype(dtype)
        assumptions.update(kwargs)
        obj = super(TypedSymbol, cls).__xnew__(cls, name, **assumptions)
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

    def __getnewargs_ex__(self):
        return (self.name, self.dtype), self.assumptions0

    @property
    def canonical(self):
        return self

    @property
    def reversed(self):
        return self

    @property
    def headers(self):
        headers = []
        try:
            if np.issubdtype(self.dtype.numpy_dtype, np.complexfloating):
                headers.append('"cuda_complex.hpp"')
        except Exception:
            pass
        try:
            if np.issubdtype(self.dtype.base_type.numpy_dtype, np.complexfloating):
                headers.append('"cuda_complex.hpp"')
        except Exception:
            pass

        return headers


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


def peel_off_type(dtype, type_to_peel_off):
    while type(dtype) is type_to_peel_off:
        dtype = dtype.base_type
    return dtype


def collate_types(types,
                  forbid_collation_to_complex=False,
                  forbid_collation_to_float=False,
                  default_float_type='float64',
                  default_int_type='int64'):
    """
    Takes a sequence of types and returns their "common type" e.g. (float, double, float) -> double
    Uses the collation rules from numpy.
    """
    if forbid_collation_to_complex:
        types = [t for t in types if not np.issubdtype(t.numpy_dtype, np.complexfloating)]
        if not types:
            return create_type(default_float_type)

    if forbid_collation_to_float:
        types = [t for t in types if not np.issubdtype(t.numpy_dtype, np.floating)]
        if not types:
            return create_type(default_int_type)

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


@memorycache_if_hashable(maxsize=2048)
def get_type_of_expression(expr,
                           default_float_type='double',
                           default_int_type='int',
                           symbol_type_dict=None):
    from pystencils.astnodes import ResolvedFieldAccess
    from pystencils.cpu.vectorization import vec_all, vec_any

    if default_float_type == 'float':
        default_float_type = 'float32'

    if not symbol_type_dict:
        symbol_type_dict = defaultdict(lambda: create_type('double'))

    get_type = partial(get_type_of_expression,
                       default_float_type=default_float_type,
                       default_int_type=default_int_type,
                       symbol_type_dict=symbol_type_dict)

    expr = sp.sympify(expr)
    if isinstance(expr, sp.Integer):
        return create_type(default_int_type)
    elif expr.is_real is False:
        return create_type((np.zeros((1,), default_float_type) * 1j).dtype)
    elif isinstance(expr, sp.Rational) or isinstance(expr, sp.Float):
        return create_type(default_float_type)
    elif isinstance(expr, ResolvedFieldAccess):
        return expr.field.dtype
    elif isinstance(expr, pystencils.field.Field.AbstractAccess):
        return expr.field.dtype
    elif isinstance(expr, TypedSymbol):
        return expr.dtype
    elif isinstance(expr, sp.Symbol):
        if symbol_type_dict:
            return symbol_type_dict[expr.name]
        else:
            raise ValueError("All symbols inside this expression have to be typed! ", str(expr))
    elif isinstance(expr, cast_func):
        return expr.args[1]
    elif isinstance(expr, (vec_any, vec_all)):
        return create_type("bool")
    elif hasattr(expr, 'func') and expr.func == sp.Piecewise:
        collated_result_type = collate_types(tuple(get_type(a[0]) for a in expr.args))
        collated_condition_type = collate_types(tuple(get_type(a[1]) for a in expr.args))
        if type(collated_condition_type) is VectorType and type(collated_result_type) is not VectorType:
            collated_result_type = VectorType(collated_result_type, width=collated_condition_type.width)
        return collated_result_type
    elif isinstance(expr, sp.Indexed):
        typed_symbol = expr.base.label
        return typed_symbol.dtype.base_type
    elif isinstance(expr, (Boolean, BooleanFunction)):
        # if any arg is of vector type return a vector boolean, else return a normal scalar boolean
        result = create_type("bool")
        vec_args = [get_type(a) for a in expr.args if isinstance(get_type(a), VectorType)]
        if vec_args:
            result = VectorType(result, width=vec_args[0].width)
        return result
    elif isinstance(expr, sp.Pow):
        base_type = get_type(expr.args[0])
        if expr.exp.is_integer:
            return base_type
        else:
            return collate_types([create_type(default_float_type), base_type])
    elif isinstance(expr, (sp.Sum, sp.Product)):
        return get_type(expr.args[0])
    elif isinstance(expr, sp.Expr):
        expr: sp.Expr
        if expr.args:
            types = tuple(get_type(a) for a in expr.args)
            # collate_types checks numpy_dtype in the special cases
            if any(not hasattr(t, 'numpy_dtype') for t in types):
                forbid_collation_to_complex = False
                forbid_collation_to_float = False
            else:
                forbid_collation_to_complex = expr.is_real is True
                forbid_collation_to_float = expr.is_integer is True
            return collate_types(
                types,
                forbid_collation_to_complex=forbid_collation_to_complex,
                forbid_collation_to_float=forbid_collation_to_float,
                default_float_type=default_float_type,
                default_int_type=default_int_type)
        else:
            if expr.is_integer:
                return create_type(default_int_type)
            else:
                return create_type(default_float_type)

    raise NotImplementedError("Could not determine type for", expr, type(expr))


sympy_version = sp.__version__.split('.')
if int(sympy_version[0]) * 100 + int(sympy_version[1]) >= 109:
    # __setstate__ would bypass the contructor, so we remove it
    sp.Number.__getstate__ = sp.Basic.__getstate__
    del sp.Basic.__getstate__

    class FunctorWithStoredKwargs:
        def __init__(self, func, **kwargs):
            self.func = func
            self.kwargs = kwargs

        def __call__(self, *args):
            return self.func(*args, **self.kwargs)

    # __reduce_ex__ would strip kwargs, so we override it
    def basic_reduce_ex(self, protocol):
        if hasattr(self, '__getnewargs_ex__'):
            args, kwargs = self.__getnewargs_ex__()
        else:
            args, kwargs = self.__getnewargs__(), {}
        if hasattr(self, '__getstate__'):
            state = self.__getstate__()
        else:
            state = None
        return FunctorWithStoredKwargs(type(self), **kwargs), args, state
    sp.Number.__reduce_ex__ = sp.Basic.__reduce_ex__
    sp.Basic.__reduce_ex__ = basic_reduce_ex


class Type(sp.Atom):
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
        elif name == 'complex64':
            return 'ComplexFloat'
        elif name == 'complex128':
            return 'ComplexDouble'
        elif name.startswith('int'):
            width = int(name[len("int"):])
            return f"int{width}_t"
        elif name.startswith('uint'):
            width = int(name[len("uint"):])
            return f"uint{width}_t"
        elif name == 'bool':
            return 'bool'
        else:
            raise NotImplementedError(f"Can map numpy to C name for {name}")

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

    def __getnewargs_ex__(self):
        return (self.numpy_dtype, self.const), {}

    @property
    def base_type(self):
        return None

    @property
    def numpy_dtype(self):
        return self._dtype

    @property
    def sympy_dtype(self):
        return getattr(sympy.codegen.ast, str(self.numpy_dtype))

    @property
    def item_size(self):
        return 1

    def is_int(self):
        return self.numpy_dtype in np.sctypes['int'] or self.numpy_dtype in np.sctypes['uint']

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
            return f"{self.base_type}[{self.width}]"
        else:
            if self.base_type == create_type("int64") or self.base_type == create_type("int32"):
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

    def __getnewargs_ex__(self):
        return (self._base_type, self.width), {}


class PointerType(Type):
    def __init__(self, base_type, const=False, restrict=True):
        self._base_type = base_type
        self.const = const
        self.restrict = restrict

    def __getnewargs__(self):
        return self.base_type, self.const, self.restrict

    def __getnewargs_ex__(self):
        return (self.base_type, self.const, self.restrict), {}

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

    def __getnewargs_ex__(self):
        return (self.numpy_dtype, self.const), {}

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


class TypedImaginaryUnit(TypedSymbol):
    def __new__(cls, *args, **kwds):
        obj = TypedImaginaryUnit.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, dtype):
        obj = super(TypedImaginaryUnit, cls).__xnew__(cls,
                                                      "_i",
                                                      dtype,
                                                      imaginary=True)
        return obj

    headers = ['"cuda_complex.hpp"']

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def __getnewargs__(self):
        return (self.dtype,)

    def __getnewargs_ex__(self):
        return (self.dtype,), {}
