from typing import Any

from .types import PsNumericType, constify
from .exceptions import PsInternalCompilerError


class PsConstant:
    __match_args__ = ("value", "dtype")

    def __init__(self, value: Any, dtype: PsNumericType | None = None):
        self._dtype: PsNumericType | None = None
        self._value = value

        if dtype is not None:
            self.apply_dtype(dtype)

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

    def apply_dtype(self, dtype: PsNumericType):
        if self._dtype is not None:
            raise PsInternalCompilerError(
                "Attempt to apply data type to already typed constant."
            )

        self._dtype = constify(dtype)
        self._value = self._dtype.create_constant(self._value)

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
