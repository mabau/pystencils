from __future__ import annotations
from abc import ABC, abstractmethod
from typing import final, TypeVar, Any, Sequence, cast
from dataclasses import dataclass
from copy import copy

import numpy as np

from .exception import PsTypeError


class PsType(ABC):
    """Base class for all pystencils types.

    Args:
        const: Const-qualification of this type
    """

    def __init__(self, const: bool = False):
        self._const = const

    @property
    def const(self) -> bool:
        return self._const

    #   -------------------------------------------------------------------------------------------
    #   Optional Info
    #   -------------------------------------------------------------------------------------------

    @property
    def required_headers(self) -> set[str]:
        """The set of header files required when this type occurs in generated code."""
        return set()

    @property
    def itemsize(self) -> int | None:
        """If this type has a valid in-memory size, return that size."""
        return None

    @property
    def numpy_dtype(self) -> np.dtype | None:
        """A np.dtype object representing this data type.

        Available both for backward compatibility and for interaction with the numpy-based runtime system.
        """
        return None

    #   -------------------------------------------------------------------------------------------
    #   Internal operations
    #   -------------------------------------------------------------------------------------------

    @abstractmethod
    def __args__(self) -> tuple[Any, ...]:
        """Arguments to this type.
        
        The tuple returned by this method is used to serialize, deserialize, and check equality of types.
        For each instantiable subclass ``MyType`` of ``PsType``, the following must hold:

        ```
        t = MyType(< arguments >)
        assert MyType(*t.__args__()) == t
        ```
        """
        pass

    def _const_string(self) -> str:
        return "const " if self._const else ""

    @abstractmethod
    def c_string(self) -> str:
        pass

    #   -------------------------------------------------------------------------------------------
    #   Dunder Methods
    #   -------------------------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        
        if type(self) is not type(other):
            return False
        
        other = cast(PsType, other)
        return self.__args__() == other.__args__()

    def __str__(self) -> str:
        return self.c_string()

    def __hash__(self) -> int:
        return hash((type(self), self.__args__()))


class PsCustomType(PsType):
    """Class to model custom types by their names.

    Args:
        name: Name of the custom type.
    """

    __match_args__ = ("name",)

    def __init__(self, name: str, const: bool = False):
        super().__init__(const)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __args__(self) -> tuple[Any, ...]:
        """
        >>> t = PsCustomType("std::vector< int >")
        >>> t == PsCustomType(*t.__args__())
        True
        """
        return (self._name,)

    def c_string(self) -> str:
        return f"{self._const_string()} {self._name}"

    def __repr__(self) -> str:
        return f"CustomType( {self.name}, const={self.const} )"


class PsDereferencableType(PsType, ABC):
    """Base class for subscriptable types.

    `PsDereferencableType` represents any type that may be dereferenced and may
    occur as the base of a subscript, that is, before the C `[]` operator.

    Args:
        base_type: The base type, which is the type of the object obtained by dereferencing.
        const: Const-qualification
    """

    __match_args__ = ("base_type",)

    def __init__(self, base_type: PsType, const: bool = False):
        super().__init__(const)
        self._base_type = base_type

    @property
    def base_type(self) -> PsType:
        return self._base_type


@final
class PsPointerType(PsDereferencableType):
    """A C pointer with arbitrary base type.

    `PsPointerType` models C pointer types to arbitrary data, with support for ``restrict``-qualified pointers.
    """

    __match_args__ = ("base_type",)

    def __init__(self, base_type: PsType, const: bool = False, restrict: bool = True):
        super().__init__(base_type, const)
        self._restrict = restrict

    def __args__(self) -> tuple[Any, ...]:
        """
        >>> t = PsPointerType(PsBoolType(), const=True)
        >>> t == PsPointerType(*t.__args__())
        True
        """
        return (self._base_type, self._const, self._restrict)

    @property
    def restrict(self) -> bool:
        return self._restrict

    def c_string(self) -> str:
        base_str = self._base_type.c_string()
        restrict_str = " RESTRICT" if self._restrict else ""
        return f"{base_str} *{restrict_str} {self._const_string()}"

    def __repr__(self) -> str:
        return f"PsPointerType( {repr(self.base_type)}, const={self.const} )"


class PsArrayType(PsDereferencableType):
    """C array type of known or unknown size."""

    def __init__(
        self, base_type: PsType, length: int | None = None, const: bool = False
    ):
        self._length = length
        super().__init__(base_type, const)

    def __args__(self) -> tuple[Any, ...]:
        """
        >>> t = PsArrayType(PsBoolType(), 13, const=True)
        >>> t == PsArrayType(*t.__args__())
        True
        """
        return (self._base_type, self._length, self._const)

    @property
    def length(self) -> int | None:
        return self._length

    def c_string(self) -> str:
        return f"{self._base_type.c_string()} [{str(self._length) if self._length is not None else ''}]"

    def __repr__(self) -> str:
        return f"PsArrayType(element_type={repr(self._base_type)}, size={self._length}, const={self._const})"


class PsStructType(PsType):
    """Named or anonymous structured data type.

    A struct type is defined by its sequence of members.
    The struct may optionally have a name, although the code generator currently does not support named structs
    and treats them the same way as anonymous structs.
    """

    @dataclass(frozen=True)
    class Member:
        name: str
        dtype: PsType

    def __init__(
        self,
        members: Sequence[PsStructType.Member | tuple[str, PsType]],
        name: str | None = None,
        const: bool = False,
    ):
        super().__init__(const=const)

        self._name = name
        self._members = tuple(
            (PsStructType.Member(m[0], m[1]) if isinstance(m, tuple) else m)
            for m in members
        )

        names: set[str] = set()
        for member in self._members:
            if member.name in names:
                raise ValueError(f"Duplicate struct member name: {member.name}")
            names.add(member.name)

    def __args__(self) -> tuple[Any, ...]:
        """
        >>> t = PsStructType([("idx", PsSignedIntegerType(32)), ("val", PsBoolType())], "sname")
        >>> t == PsStructType(*t.__args__())
        True
        """
        return (self._members, self._name, self._const)

    @property
    def members(self) -> tuple[PsStructType.Member, ...]:
        return self._members

    def find_member(self, member_name: str) -> PsStructType.Member | None:
        """Find a member by name"""
        for m in self._members:
            if m.name == member_name:
                return m
        return None

    def get_member(self, member_name: str) -> PsStructType.Member:
        m = self.find_member(member_name)
        if m is None:
            raise KeyError(f"No struct member with name {member_name}")
        return m

    @property
    def name(self) -> str:
        if self._name is None:
            raise PsTypeError("Cannot retrieve name from anonymous struct type")
        return self._name

    @property
    def anonymous(self) -> bool:
        return self._name is None

    @property
    def numpy_dtype(self) -> np.dtype:
        members = [(m.name, m.dtype.numpy_dtype) for m in self._members]
        return np.dtype(members)

    @property
    def itemsize(self) -> int:
        return self.numpy_dtype.itemsize

    def c_string(self) -> str:
        if self._name is None:
            raise PsTypeError("Cannot retrieve C string for anonymous struct type")
        return self._name

    def __str__(self) -> str:
        if self._name is None:
            return "<anonymous>"
        else:
            return self._name

    def __repr__(self) -> str:
        members = ", ".join(f"{m.dtype} {m.name}" for m in self._members)
        name = "<anonymous>" if self.anonymous else f"name={self._name}"
        return f"PsStructType( [{members}], {name}, const={self.const} )"


class PsNumericType(PsType, ABC):
    """Numeric data type, i.e. any type that may occur inside arithmetic-logical expressions.

    **Constants**

    Every numeric type has to act as a factory for compile-time constants of that type.
    The `PsConstant` class relies on `create_constant` to instantiate constants
    of a given numeric type. The object returned by `create_constant` must implement the
    necessary arithmetic operations, and its arithmetic behaviour must match the given type.

    `create_constant` should fail whenever its input cannot safely be interpreted as the given
    type.
    """

    @abstractmethod
    def create_constant(self, value: Any) -> Any:
        """
        Create the internal representation of a constant with this type.

        Raises:
            PsTypeError: If the given value cannot be interpreted in this type.
        """

    @abstractmethod
    def is_int(self) -> bool:
        pass

    @abstractmethod
    def is_sint(self) -> bool:
        pass

    @abstractmethod
    def is_uint(self) -> bool:
        pass

    @abstractmethod
    def is_float(self) -> bool:
        pass

    @abstractmethod
    def is_bool(self) -> bool:
        pass


class PsScalarType(PsNumericType, ABC):
    """Scalar numeric type."""

    @abstractmethod
    def create_literal(self, value: Any) -> str:
        """Create a C numerical literal for a constant of this type.

        Raises:
            PsTypeError: If the given value's type is not the numeric type's compiler-internal representation.
        """

    @property
    @abstractmethod
    def width(self) -> int:
        """Return this type's width in bits."""

    def is_int(self) -> bool:
        return isinstance(self, PsIntegerType)

    def is_sint(self) -> bool:
        return isinstance(self, PsIntegerType) and self.signed

    def is_uint(self) -> bool:
        return isinstance(self, PsIntegerType) and not self.signed

    def is_float(self) -> bool:
        return isinstance(self, PsIeeeFloatType)

    def is_bool(self) -> bool:
        return isinstance(self, PsBoolType)


class PsVectorType(PsNumericType):
    """Packed vector of numeric type.

    Args:
        element_type: Underlying scalar data type
        num_entries: Number of entries in the vector
    """

    def __init__(
        self, scalar_type: PsScalarType, vector_entries: int, const: bool = False
    ):
        super().__init__(const)
        self._vector_entries = vector_entries
        self._scalar_type = constify(scalar_type) if const else deconstify(scalar_type)

    def __args__(self) -> tuple[Any, ...]:
        """
        >>> t = PsVectorType(PsBoolType(), 8, True)
        >>> t == PsVectorType(*t.__args__())
        True
        """
        return (self._scalar_type, self._vector_entries, self._const)

    @property
    def scalar_type(self) -> PsScalarType:
        return self._scalar_type

    @property
    def vector_entries(self) -> int:
        return self._vector_entries

    @property
    def width(self) -> int:
        return self._scalar_type.width * self._vector_entries

    def is_int(self) -> bool:
        return self._scalar_type.is_int()

    def is_sint(self) -> bool:
        return self._scalar_type.is_sint()

    def is_uint(self) -> bool:
        return self._scalar_type.is_uint()

    def is_float(self) -> bool:
        return self._scalar_type.is_float()

    def is_bool(self) -> bool:
        return self._scalar_type.is_bool()

    @property
    def itemsize(self) -> int | None:
        if self._scalar_type.itemsize is None:
            return None
        else:
            return self._vector_entries * self._scalar_type.itemsize

    @property
    def numpy_dtype(self):
        return np.dtype((self._scalar_type.numpy_dtype, (self._vector_entries,)))

    def create_constant(self, value: Any) -> Any:
        if (
            isinstance(value, np.ndarray)
            and value.dtype == self.scalar_type.numpy_dtype
            and value.shape == (self._vector_entries,)
        ):
            return value.copy()

        element = self._scalar_type.create_constant(value)
        return np.array(
            [element] * self._vector_entries, dtype=self.scalar_type.numpy_dtype
        )

    def c_string(self) -> str:
        raise PsTypeError("Cannot retrieve C type string for generic vector types.")

    def __str__(self) -> str:
        return f"vector[{self._scalar_type}, {self._vector_entries}]"

    def __repr__(self) -> str:
        return (
            f"PsVectorType( scalar_type={repr(self._scalar_type)}, "
            f"vector_width={self._vector_entries}, const={self.const} )"
        )


class PsBoolType(PsScalarType):
    """Boolean type."""

    NUMPY_TYPE = np.bool_

    def __init__(self, const: bool = False):
        super().__init__(const)

    def __args__(self) -> tuple[Any, ...]:
        """
        >>> t = PsBoolType(True)
        >>> t == PsBoolType(*t.__args__())
        True
        """
        return (self._const,)

    @property
    def width(self) -> int:
        return 8

    @property
    def itemsize(self) -> int:
        return self.width // 8

    @property
    def numpy_dtype(self) -> np.dtype | None:
        return np.dtype(PsBoolType.NUMPY_TYPE)

    def create_literal(self, value: Any) -> str:
        if not isinstance(value, self.NUMPY_TYPE):
            raise PsTypeError(f"Given value {value} is not of required type {self.NUMPY_TYPE}")

        if value == np.True_:
            return "true"
        elif value == np.False_:
            return "false"
        else:
            raise PsTypeError(f"Cannot create boolean literal from {value}")

    def create_constant(self, value: Any) -> Any:
        if value in (1, True, np.True_):
            return np.True_
        elif value in (0, False, np.False_):
            return np.False_
        else:
            raise PsTypeError(f"Cannot create boolean constant from value {value}")

    def c_string(self) -> str:
        return "bool"
    

class PsIntegerType(PsScalarType, ABC):
    """Signed and unsigned integer types.

    `PsIntegerType` cannot be instantiated on its own, but only through `PsSignedIntegerType`
    and `PsUnsignedIntegerType`. This distinction is meant mostly to help in pattern matching.
    """

    __match_args__ = ("width",)

    SUPPORTED_WIDTHS = (8, 16, 32, 64)
    NUMPY_TYPES: dict[int, type] = dict()

    def __init__(self, width: int, signed: bool = True, const: bool = False):
        if width not in self.SUPPORTED_WIDTHS:
            raise ValueError(
                f"Invalid integer width; must be one of {self.SUPPORTED_WIDTHS}."
            )

        super().__init__(const)

        self._width = width
        self._signed = signed

    @property
    def width(self) -> int:
        return self._width

    @property
    def signed(self) -> bool:
        return self._signed

    @property
    def itemsize(self) -> int:
        return self.width // 8

    @property
    def numpy_dtype(self) -> np.dtype | None:
        return np.dtype(self.NUMPY_TYPES[self._width])

    def create_literal(self, value: Any) -> str:
        np_dtype = self.NUMPY_TYPES[self._width]
        if not isinstance(value, np_dtype):
            raise PsTypeError(f"Given value {value} is not of required type {np_dtype}")
        unsigned_suffix = "" if self.signed else "u"
        #   TODO: cast literal to correct type?
        return str(value) + unsigned_suffix
    
    def create_constant(self, value: Any) -> Any:
        np_type = self.NUMPY_TYPES[self._width]

        if isinstance(value, (int, np.integer)):
            iinfo = np.iinfo(np_type)  # type: ignore
            if value < iinfo.min or value > iinfo.max:
                raise PsTypeError(f"Could not interpret {value} as {self}: Value is out of bounds.")
            return np_type(value)

        raise PsTypeError(f"Could not interpret {value} as {repr(self)}")
    
    def c_string(self) -> str:
        prefix = "" if self._signed else "u"
        return f"{self._const_string()}{prefix}int{self._width}_t"

    def __repr__(self) -> str:
        return f"PsIntegerType( width={self.width}, signed={self.signed}, const={self.const} )"


@final
class PsSignedIntegerType(PsIntegerType):
    """Signed integer types."""

    __match_args__ = ("width",)

    NUMPY_TYPES = {
        8: np.int8,
        16: np.int16,
        32: np.int32,
        64: np.int64,
    }

    def __init__(self, width: int, const: bool = False):
        super().__init__(width, True, const)

    def __args__(self) -> tuple[Any, ...]:
        """
        >>> t = PsSignedIntegerType(32, True)
        >>> t == PsSignedIntegerType(*t.__args__())
        True
        """
        return (self._width, self._const)


@final
class PsUnsignedIntegerType(PsIntegerType):
    """Unsigned integer types."""

    __match_args__ = ("width",)

    NUMPY_TYPES = {
        8: np.uint8,
        16: np.uint16,
        32: np.uint32,
        64: np.uint64,
    }

    def __init__(self, width: int, const: bool = False):
        super().__init__(width, False, const)

    def __args__(self) -> tuple[Any, ...]:
        """
        >>> t = PsUnsignedIntegerType(32, True)
        >>> t == PsUnsignedIntegerType(*t.__args__())
        True
        """
        return (self._width, self._const)


@final
class PsIeeeFloatType(PsScalarType):
    """IEEE-754 floating point data types"""

    __match_args__ = ("width",)

    SUPPORTED_WIDTHS = (16, 32, 64)

    NUMPY_TYPES = {
        16: np.float16,
        32: np.float32,
        64: np.float64,
    }

    def __init__(self, width: int, const: bool = False):
        if width not in self.SUPPORTED_WIDTHS:
            raise ValueError(
                f"Invalid integer width {width}; must be one of {self.SUPPORTED_WIDTHS}."
            )

        super().__init__(const)
        self._width = width

    def __args__(self) -> tuple[Any, ...]:
        """
        >>> t = PsIeeeFloatType(32, True)
        >>> t == PsIeeeFloatType(*t.__args__())
        True
        """
        return (self._width, self._const)

    @property
    def width(self) -> int:
        return self._width

    @property
    def itemsize(self) -> int:
        return self.width // 8

    @property
    def numpy_dtype(self) -> np.dtype | None:
        return np.dtype(self.NUMPY_TYPES[self._width])

    @property
    def required_headers(self) -> set[str]:
        if self._width == 16:
            return {'"half_precision.h"'}
        else:
            return set()

    def create_literal(self, value: Any) -> str:
        np_dtype = self.NUMPY_TYPES[self._width]
        if not isinstance(value, np_dtype):
            raise PsTypeError(f"Given value {value} is not of required type {np_dtype}")

        match self.width:
            case 16:
                return f"((half) {value})"  # see include/half_precision.h
            case 32:
                return f"{value}f"
            case 64:
                return str(value)
            case _:
                assert False, "unreachable code"

    def create_constant(self, value: Any) -> Any:
        np_type = self.NUMPY_TYPES[self._width]

        if isinstance(value, (int, float, np.floating)):
            finfo = np.finfo(np_type)  # type: ignore
            if value < finfo.min or value > finfo.max:
                raise PsTypeError(f"Could not interpret {value} as {self}: Value is out of bounds.")
            return np_type(value)

        raise PsTypeError(f"Could not interpret {value} as {repr(self)}")

    def c_string(self) -> str:
        match self._width:
            case 16:
                return f"{self._const_string()}half"
            case 32:
                return f"{self._const_string()}float"
            case 64:
                return f"{self._const_string()}double"
            case _:
                assert False, "unreachable code"

    def __repr__(self) -> str:
        return f"PsIeeeFloatType( width={self.width}, const={self.const} )"


T = TypeVar("T", bound=PsType)


def constify(t: T) -> T:
    """Adds the const qualifier to a given type."""
    if not t.const:
        t_copy = copy(t)
        t_copy._const = True
        return t_copy
    else:
        return t


def deconstify(t: T) -> T:
    """Removes the const qualifier from a given type."""
    if t.const:
        t_copy = copy(t)
        t_copy._const = False
        return t_copy
    else:
        return t
