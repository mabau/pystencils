from abc import abstractmethod
from typing import Union

import numpy as np
import sympy as sp


class AbstractType(sp.Atom):
    # TODO: Is it necessary to ineherit from sp.Atom?
    def __new__(cls, *args, **kwargs):
        return sp.Basic.__new__(cls)

    def _sympystr(self, *args, **kwargs):
        return str(self)

    @property
    @abstractmethod
    def base_type(self) -> Union[None, 'BasicType']:
        """
        Returns: Returns BasicType of a Vector or Pointer type, None otherwise
        """
        pass

    @property
    @abstractmethod
    def item_size(self) -> int:
        """
        Returns: Number of items.
        E.g. width * item_size(basic_type) in vector's case, or simple numpy itemsize in Struct's case.
        """
        pass


def is_supported_type(dtype: np.dtype):
    scalar = dtype.type
    c = np.issctype(dtype)
    subclass = issubclass(scalar, np.floating) or issubclass(scalar, np.integer) or issubclass(scalar, np.bool_)
    additional_checks = dtype.fields is None and dtype.hasobject is False and dtype.subdtype is None
    return c and subclass and additional_checks


def numpy_name_to_c(name: str) -> str:
    """
    Converts a np.dtype.name into a C type
    Args:
        name: np.dtype.name string
    Returns:
        type as a C string
    """
    if name == 'float64':
        return 'double'
    elif name == 'float32':
        return 'float'
    elif name == 'float16' or name == 'half':
        return 'half'
    elif name.startswith('int'):
        width = int(name[len("int"):])
        return f"int{width}_t"
    elif name.startswith('uint'):
        width = int(name[len("uint"):])
        return f"uint{width}_t"
    elif name == 'bool':
        return 'bool'
    else:
        raise NotImplementedError(f"Can't map numpy to C name for {name}")


def create_type(specification: Union[type, AbstractType, str]) -> AbstractType:
    # TODO: Deprecated Use the constructor of BasicType or StructType instead
    """Creates a subclass of Type according to a string or an object of subclass Type.

    Args:
        specification: Type object, or a string

    Returns:
        Type object, or a new Type object parsed from the string
    """
    if isinstance(specification, AbstractType):
        return specification
    else:
        numpy_dtype = np.dtype(specification)
        if numpy_dtype.fields is None:
            return BasicType(numpy_dtype, const=False)
        else:
            return StructType(numpy_dtype, const=False)


def get_base_type(data_type):
    """
    Returns the BasicType of a Pointer or a Vector
    """
    while data_type.base_type is not None:
        data_type = data_type.base_type
    return data_type


class BasicType(AbstractType):
    """
    BasicType is defined with a const qualifier and a np.dtype.
    """

    def __init__(self, dtype: Union[type, 'BasicType', str], const: bool = False):
        if isinstance(dtype, BasicType):
            self.numpy_dtype = dtype.numpy_dtype
            self.const = dtype.const
        else:
            self.numpy_dtype = np.dtype(dtype)
            self.const = const
        assert is_supported_type(self.numpy_dtype), f'Type {self.numpy_dtype} is currently not supported!'

    def __getnewargs__(self):
        return self.numpy_dtype, self.const

    def __getnewargs_ex__(self):
        return (self.numpy_dtype, self.const), {}

    @property
    def base_type(self):
        return None

    @property
    def item_size(self):  # TODO: Do we want self.numpy_type.itemsize????
        return 1

    def is_float(self):
        return issubclass(self.numpy_dtype.type, np.floating)

    def is_half(self):
        return issubclass(self.numpy_dtype.type, np.half)

    def is_int(self):
        return issubclass(self.numpy_dtype.type, np.integer)

    def is_uint(self):
        return issubclass(self.numpy_dtype.type, np.unsignedinteger)

    def is_sint(self):
        return issubclass(self.numpy_dtype.type, np.signedinteger)

    def is_bool(self):
        return issubclass(self.numpy_dtype.type, np.bool_)

    def dtype_eq(self, other):
        if not isinstance(other, BasicType):
            return False
        else:
            return self.numpy_dtype == other.numpy_dtype

    @property
    def c_name(self) -> str:
        return numpy_name_to_c(self.numpy_dtype.name)

    def __str__(self):
        return f'{self.c_name}{" const" if self.const else ""}'

    def __repr__(self):
        return f'BasicType( {str(self)} )'

    def _repr_html_(self):
        return f'BasicType( {str(self)} )'

    def __eq__(self, other):
        return self.dtype_eq(other) and self.const == other.const

    def __hash__(self):
        return hash(str(self))


class VectorType(AbstractType):
    """
    VectorType consists of a BasicType and a width.
    """
    instruction_set = None

    def __init__(self, base_type: BasicType, width: int):
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
            # TODO VectorizationRevamp: this seems super weird. the instruction_set should know how to print a type out!
            # TODO VectorizationRevamp: this is error prone. base_type could be cons=True. Use dtype instead
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


class PointerType(AbstractType):
    def __init__(self, base_type: BasicType, const: bool = False, restrict: bool = True, double_pointer: bool = False):
        self._base_type = base_type
        self.const = const
        self.restrict = restrict
        self.double_pointer = double_pointer

    def __getnewargs__(self):
        return self.base_type, self.const, self.restrict, self.double_pointer

    def __getnewargs_ex__(self):
        return (self.base_type, self.const, self.restrict, self.double_pointer), {}

    @property
    def alias(self):
        return not self.restrict

    @property
    def base_type(self):
        return self._base_type

    @property
    def item_size(self):
        if self.double_pointer:
            raise NotImplementedError("The item_size for double_pointer is not implemented")
        else:
            return self.base_type.item_size

    def __eq__(self, other):
        if not isinstance(other, PointerType):
            return False
        else:
            own = (self.base_type, self.const, self.restrict, self.double_pointer)
            return own == (other.base_type, other.const, other.restrict, other.double_pointer)

    def __str__(self):
        restrict_str = "RESTRICT" if self.restrict else ""
        const_str = "const" if self.const else ""
        if self.double_pointer:
            return f'{str(self.base_type)} ** {restrict_str} {const_str}'
        else:
            return f'{str(self.base_type)} * {restrict_str} {const_str}'

    def __repr__(self):
        return str(self)

    def _repr_html_(self):
        return str(self)

    def __hash__(self):
        return hash((self._base_type, self.const, self.restrict, self.double_pointer))


class StructType(AbstractType):
    """
    A list of types (with C offsets).
    It is implemented with uint8_t and casts to the correct datatype.
    """
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

    def _repr_html_(self):
        return str(self)

    def __hash__(self):
        return hash((self.numpy_dtype, self.const))


def assumptions_from_dtype(dtype: Union[BasicType, np.dtype]):
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
    except Exception:  # TODO this is dirty
        pass

    return assumptions


class TypedSymbol(sp.Symbol):
    def __new__(cls, *args, **kwds):
        obj = TypedSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, name, dtype, **kwargs):  # TODO does not match signature of sp.Symbol???
        # TODO: also Symbol should be allowed  ---> see sympy Variable
        assumptions = assumptions_from_dtype(dtype)
        assumptions.update(kwargs)
        obj = super(TypedSymbol, cls).__xnew__(cls, name, **assumptions)
        try:
            obj.numpy_dtype = create_type(dtype)
        except (TypeError, ValueError):
            # on error keep the string
            obj.numpy_dtype = dtype
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(sp.core.cacheit(__new_stage2__))

    @property
    def dtype(self):
        return self.numpy_dtype

    def _hashable_content(self):
        return super()._hashable_content(), hash(self.numpy_dtype)

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


SHAPE_DTYPE = BasicType('int64', const=True)
STRIDE_DTYPE = BasicType('int64', const=True)


class FieldStrideSymbol(TypedSymbol):
    """Sympy symbol representing the stride value of a field in a specific coordinate."""
    def __new__(cls, *args, **kwds):
        obj = FieldStrideSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, field_name, coordinate):
        name = f"_stride_{field_name}_{coordinate}"
        obj = super(FieldStrideSymbol, cls).__xnew__(cls, name, STRIDE_DTYPE, positive=True)
        obj.field_name = field_name
        obj.coordinate = coordinate
        return obj

    def __getnewargs__(self):
        return self.field_name, self.coordinate

    def __getnewargs_ex__(self):
        return (self.field_name, self.coordinate), {}

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(sp.core.cacheit(__new_stage2__))

    def _hashable_content(self):
        return super()._hashable_content(), self.coordinate, self.field_name


class FieldShapeSymbol(TypedSymbol):
    """Sympy symbol representing the shape value of a sequence of fields. In a kernel iterating over multiple fields
    there is only one set of `FieldShapeSymbol`s since all the fields have to be of equal size."""
    def __new__(cls, *args, **kwds):
        obj = FieldShapeSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, field_names, coordinate):
        names = "_".join([field_name for field_name in field_names])
        name = f"_size_{names}_{coordinate}"
        obj = super(FieldShapeSymbol, cls).__xnew__(cls, name, SHAPE_DTYPE, positive=True)
        obj.field_names = tuple(field_names)
        obj.coordinate = coordinate
        return obj

    def __getnewargs__(self):
        return self.field_names, self.coordinate

    def __getnewargs_ex__(self):
        return (self.field_names, self.coordinate), {}

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(sp.core.cacheit(__new_stage2__))

    def _hashable_content(self):
        return super()._hashable_content(), self.coordinate, self.field_names


class FieldPointerSymbol(TypedSymbol):
    """Sympy symbol representing the pointer to the beginning of the field data."""
    def __new__(cls, *args, **kwds):
        obj = FieldPointerSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, field_name, field_dtype, const):
        name = f"_data_{field_name}"
        dtype = PointerType(get_base_type(field_dtype), const=const, restrict=True)
        obj = super(FieldPointerSymbol, cls).__xnew__(cls, name, dtype)
        obj.field_name = field_name
        return obj

    def __getnewargs__(self):
        return self.field_name, self.dtype, self.dtype.const

    def __getnewargs_ex__(self):
        return (self.field_name, self.dtype, self.dtype.const), {}

    def _hashable_content(self):
        return super()._hashable_content(), self.field_name

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(sp.core.cacheit(__new_stage2__))


class CastFunc(sp.Function):
    """
    CastFunc is used in order to introduce static casts. They are especially useful as a way to signal what type
    a certain node should have, if it is impossible to add a type to a node, e.g. a sp.Number.
    """
    is_Atom = True

    def __new__(cls, *args, **kwargs):
        if len(args) != 2:
            pass
        expr, dtype, *other_args = args

        # If we have two consecutive casts, throw the inner one away.
        # This optimisation is only available for simple casts. Thus the == is intended here!
        if expr.__class__ == CastFunc:
            expr = expr.args[0]
        if not isinstance(dtype, AbstractType):
            dtype = BasicType(dtype)
        # to work in conditions of sp.Piecewise cast_func has to be of type Boolean as well
        # however, a cast_function should only be a boolean if its argument is a boolean, otherwise this leads
        # to problems when for example comparing cast_func's for equality
        #
        # lhs = bitwise_and(a, cast_func(1, 'int'))
        # rhs = cast_func(0, 'int')
        # print( sp.Ne(lhs, rhs) ) # would give true if all cast_funcs are booleans
        # -> thus a separate class boolean_cast_func is introduced
        if (isinstance(expr, sp.logic.boolalg.Boolean) and
                (not isinstance(expr, TypedSymbol) or expr.dtype == BasicType('bool'))):
            cls = BooleanCastFunc

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

    @property
    def dtype(self):
        return self.args[1]

    @property
    def expr(self):
        return self.args[0]

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
            return np.issubdtype(self.dtype.numpy_dtype, np.integer) or np.issubdtype(self.dtype.numpy_dtype,
                                                                                      np.floating) or super().is_real
        else:
            return super().is_real


class BooleanCastFunc(CastFunc, sp.logic.boolalg.Boolean):
    # TODO: documentation
    pass


class VectorMemoryAccess(CastFunc):
    """
    Special memory access for vectorized kernel.
    Arguments: read/write expression, type, aligned, non-temporal, mask (or none), stride
    """
    nargs = (6,)


class ReinterpretCastFunc(CastFunc):
    """
    Reinterpret cast is necessary for the StructType
    """
    pass


class PointerArithmeticFunc(sp.Function, sp.logic.boolalg.Boolean):
    # TODO: documentation, or deprecate!
    @property
    def canonical(self):
        if hasattr(self.args[0], 'canonical'):
            return self.args[0].canonical
        else:
            raise NotImplementedError()



