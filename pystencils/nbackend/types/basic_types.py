from __future__ import annotations
from abc import ABC, abstractmethod
from typing import final, TypeVar, Any
from copy import copy

import numpy as np

from .exception import PsTypeError


class PsAbstractType(ABC):
    """Base class for all pystencils types.

    Implementation Notes
    ====================

    **Type Equality:** Subclasses must implement `__eq__`, but may rely on `_base_equal` to implement
    type equality checks.
    """

    def __init__(self, const: bool = False):
        """
        Args:
            name: Name of this type
            const: Const-qualification of this type
        """
        self._const = const

    @property
    def const(self) -> bool:
        return self._const

    #   -------------------------------------------------------------------------------------------
    #   Internal virtual operations
    #   -------------------------------------------------------------------------------------------

    def _base_equal(self, other: PsAbstractType) -> bool:
        return type(self) is type(other) and self._const == other._const

    def _const_string(self) -> str:
        return "const" if self._const else ""

    @abstractmethod
    def _c_string(self) -> str:
        ...

    #   -------------------------------------------------------------------------------------------
    #   Dunder Methods
    #   -------------------------------------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    def __str__(self) -> str:
        return self._c_string()

    @abstractmethod
    def __hash__(self) -> int:
        ...


class PsCustomType(PsAbstractType):
    """Class to model custom types by their names."""

    __match_args__ = ("name",)

    def __init__(self, name: str, const: bool = False):
        super().__init__(const)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsCustomType):
            return False
        return self._base_equal(other) and self._name == other._name

    def __hash__(self) -> int:
        return hash(("PsCustomType", self._name, self._const))

    def _c_string(self) -> str:
        return f"{self._const_string()} {self._name}"

    def __repr__(self) -> str:
        return f"CustomType( {self.name}, const={self.const} )"


@final
class PsPointerType(PsAbstractType):
    """Class to model C pointer types."""

    __match_args__ = ("base_type",)

    def __init__(
        self, base_type: PsAbstractType, const: bool = False, restrict: bool = True
    ):
        super().__init__(const)
        self._base_type = base_type
        self._restrict = restrict

    @property
    def base_type(self) -> PsAbstractType:
        return self._base_type

    @property
    def restrict(self) -> bool:
        return self._restrict

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsPointerType):
            return False
        return self._base_equal(other) and self._base_type == other._base_type

    def __hash__(self) -> int:
        return hash(("PsPointerType", self._base_type, self._restrict, self._const))

    def _c_string(self) -> str:
        base_str = self._base_type._c_string()
        return f"{base_str} * {self._const_string()}"

    def __repr__(self) -> str:
        return f"PsPointerType( {repr(self.base_type)}, const={self.const} )"


class PsNumericType(PsAbstractType, ABC):
    """Class to model numeric types, which are all types that may occur at the top-level inside
    arithmetic-logical expressions.

    Constants
    ---------

    Every numeric type has to act as a factory for compile-time constants of that type.
    The `PsTypedConstant` class relies on `create_constant` to instantiate constants
    of a given numeric type. The object returned by `create_constant` must implement the
    necessary arithmetic operations, and its arithmetic behaviour must match the given type.

    `create_constant` should fail whenever its input cannot safely be interpreted as the given
    type. As for which interpretations are considered 'safe', it should be as restrictive as possible.
    However, `create_constant` must *never* fail for the literals `0`, `1` and `-1`.
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
        ...

    @abstractmethod
    def is_sint(self) -> bool:
        ...

    @abstractmethod
    def is_uint(self) -> bool:
        ...

    @abstractmethod
    def is_float(self) -> bool:
        ...


class PsScalarType(PsNumericType, ABC):
    """Class to model scalar numeric types."""

    def is_int(self) -> bool:
        return isinstance(self, PsIntegerType)

    def is_sint(self) -> bool:
        return isinstance(self, PsIntegerType) and self.signed

    def is_uint(self) -> bool:
        return isinstance(self, PsIntegerType) and not self.signed

    def is_float(self) -> bool:
        return isinstance(self, PsIeeeFloatType)


class PsIntegerType(PsScalarType, ABC):
    """Class to model signed and unsigned integer types.

    `PsIntegerType` cannot be instantiated on its own, but only through `PsSignedIntegerType`
    and `PsUnsignedIntegerType`. This distinction is meant mostly to help in pattern matching.
    """

    __match_args__ = ("width",)

    SUPPORTED_WIDTHS = (8, 16, 32, 64)

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsIntegerType):
            return False

        return (
            self._base_equal(other)
            and self._width == other._width
            and self._signed == other._signed
        )

    def __hash__(self) -> int:
        return hash(("PsIntegerType", self._width, self._signed, self._const))

    def _c_string(self) -> str:
        prefix = "" if self._signed else "u"
        return f"{self._const_string()} {prefix}int{self._width}_t"

    def __repr__(self) -> str:
        return f"PsIntegerType( width={self.width}, signed={self.signed}, const={self.const} )"


@final
class PsSignedIntegerType(PsIntegerType):
    """Class to model signed integers."""

    __match_args__ = ("width",)

    NUMPY_TYPES = {
        8: np.int8,
        16: np.int16,
        32: np.int32,
        64: np.int64,
    }

    def __init__(self, width: int, const: bool = False):
        super().__init__(width, True, const)

    def create_constant(self, value: Any) -> Any:
        np_type = self.NUMPY_TYPES[self._width]

        if isinstance(value, int):
            return np_type(value)

        if isinstance(value, np_type):
            return value

        raise PsTypeError(f"Could not interpret {value} as {repr(self)}")


@final
class PsUnsignedIntegerType(PsIntegerType):
    """Class to model unsigned integers."""

    __match_args__ = ("width",)

    NUMPY_TYPES = {
        8: np.uint8,
        16: np.uint16,
        32: np.uint32,
        64: np.uint64,
    }

    def __init__(self, width: int, const: bool = False):
        super().__init__(width, False, const)

    def create_constant(self, value: Any) -> Any:
        np_type = self.NUMPY_TYPES[self._width]

        if isinstance(value, int) and value >= 0:
            return np_type(value)

        if isinstance(value, np_type):
            return value

        raise PsTypeError(f"Could not interpret {value} as {repr(self)}")


@final
class PsIeeeFloatType(PsScalarType):
    """Class to model IEEE-754 floating point data types"""

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
                f"Invalid integer width; must be one of {self.SUPPORTED_WIDTHS}."
            )

        super().__init__(const)
        self._width = width

    @property
    def width(self) -> int:
        return self._width

    def create_constant(self, value: Any) -> Any:
        np_type = self.NUMPY_TYPES[self._width]

        if isinstance(value, int) and value in (0, 1, -1):
            return np_type(value)

        if isinstance(value, float):
            return np_type(value)

        if isinstance(value, np_type):
            return value

        raise PsTypeError(f"Could not interpret {value} as {repr(self)}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsIeeeFloatType):
            return False
        return self._base_equal(other) and self._width == other._width

    def __hash__(self) -> int:
        return hash(("PsIeeeFloatType", self._width, self._const))

    def _c_string(self) -> str:
        match self._width:
            case 32:
                return f"{self._const_string()} float"
            case 64:
                return f"{self._const_string()} double"
            case _:
                assert False, "unreachable code"

    def __repr__(self) -> str:
        return f"PsIeeeFloatType( width={self.width}, const={self.const} )"


T = TypeVar("T", bound=PsAbstractType)


def constify(t: T) -> T:
    """Adds the const qualifier to a given type."""
    t_copy = copy(t)
    t_copy._const = True
    return t_copy


def deconstify(t: T) -> T:
    """Removes the const qualifier from a given type."""
    t_copy = copy(t)
    t_copy._const = False
    return t_copy
