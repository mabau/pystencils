from ..types import PsType, PsTypeError
from .exceptions import PsInternalCompilerError


class PsSymbol:
    """A mutable symbol with name and data type.

    Do not create objects of this class directly unless you know what you are doing;
    instead obtain them from a `KernelCreationContext` through `KernelCreationContext.get_symbol`.
    This way, the context can keep track of all symbols used in the translation run,
    and uniqueness of symbols is ensured.
    """

    __match_args__ = ("name", "dtype")

    def __init__(self, name: str, dtype: PsType | None = None):
        self._name = name
        self._dtype = dtype

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> PsType | None:
        return self._dtype

    @dtype.setter
    def dtype(self, value: PsType):
        self._dtype = value

    def apply_dtype(self, dtype: PsType):
        """Apply the given data type to this symbol,
        raising a TypeError if it conflicts with a previously set data type."""

        if self._dtype is not None and self._dtype != dtype:
            raise PsTypeError(
                f"Incompatible symbol data types: {self._dtype} and {dtype}"
            )

        self._dtype = dtype

    def get_dtype(self) -> PsType:
        if self._dtype is None:
            raise PsInternalCompilerError("Symbol had no type assigned yet")
        return self._dtype

    def __str__(self) -> str:
        dtype_str = "<untyped>" if self._dtype is None else str(self._dtype)
        return f"{self._name}: {dtype_str}"

    def __repr__(self) -> str:
        return f"PsSymbol({self._name}, {self._dtype})"
