from __future__ import annotations
from typing import Any

from ..types import PsNumericType, constify
from .exceptions import PsInternalCompilerError


class PsConstant:
    """Type-safe representation of typed numerical constants.
    
    This class models constants in the backend representation of kernels.
    A constant may be *untyped*, in which case its ``value`` may be any Python object.
    
    If the constant is *typed* (i.e. its ``dtype`` is not ``None``), its data type is used
    to check the validity of its ``value`` and to convert it into the type's internal representation.

    Instances of `PsConstant` are immutable.

    Args:
        value: The constant's value
        dtype: The constant's data type, or ``None`` if untyped.
    """

    __match_args__ = ("value", "dtype")

    def __init__(self, value: Any, dtype: PsNumericType | None = None):
        self._dtype: PsNumericType | None = None
        self._value = value

        if dtype is not None:
            self._dtype = constify(dtype)
            self._value = self._dtype.create_constant(self._value)
        else:
            self._dtype = None
            self._value = value

    def interpret_as(self, dtype: PsNumericType) -> PsConstant:
        """Interprets this *untyped* constant with the given data type.
        
        If this constant is already typed, raises an error.
        """
        if self._dtype is not None:
            raise PsInternalCompilerError(
                f"Cannot interpret already typed constant {self} with type {dtype}"
            )
        
        return PsConstant(self._value, dtype)
    
    def reinterpret_as(self, dtype: PsNumericType) -> PsConstant:
        """Reinterprets this constant with the given data type.
        
        Other than `interpret_as`, this method also works on typed constants.
        """
        return PsConstant(self._value, dtype)

    @property
    def value(self) -> Any:
        return self._value

    @property
    def dtype(self) -> PsNumericType | None:
        return self._dtype

    def get_dtype(self) -> PsNumericType:
        if self._dtype is None:
            raise PsInternalCompilerError("Data type of constant was not set.")
        return self._dtype

    def __str__(self) -> str:
        type_str = "<untyped>" if self._dtype is None else str(self._dtype)
        return f"{str(self._value)}: {type_str}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash((self._dtype, self._value))

    def __eq__(self, other) -> bool:
        if not isinstance(other, PsConstant):
            return False

        return (self._value, self._dtype) == (other._value, other._dtype)
