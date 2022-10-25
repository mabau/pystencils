from abc import abstractmethod
from typing import Union

import numpy as np
import sympy as sp


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


class BasicType(AbstractType):
    """
    BasicType is defined with a const qualifier and a np.dtype.
    """

    def __init__(self, dtype: Union[np.dtype, 'BasicType', str], const: bool = False):
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
    def __init__(self, base_type: BasicType, const: bool = False, restrict: bool = True):
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
        return f'{str(self.base_type)} * {"RESTRICT " if self.restrict else "" }{"const" if self.const else ""}'

    def __repr__(self):
        return str(self)

    def _repr_html_(self):
        return str(self)

    def __hash__(self):
        return hash((self._base_type, self.const, self.restrict))


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


def create_type(specification: Union[np.dtype, AbstractType, str]) -> AbstractType:
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
