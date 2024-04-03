from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TypeVar, Any, cast
import numpy as np


class PsTypeMeta(ABCMeta):

    _instances: dict[Any, PsType] = dict()

    def __call__(cls, *args: Any, const: bool = False, **kwargs: Any) -> Any:
        obj = super(PsTypeMeta, cls).__call__(*args, const=const, **kwargs)
        canonical_args = obj.__args__()
        key = (cls, canonical_args, const)

        if key in cls._instances:
            obj = cls._instances[key]
        else:
            cls._instances[key] = obj

        return obj


class PsType(metaclass=PsTypeMeta):
    """Base class for all pystencils types.

    Args:
        const: Const-qualification of this type

    **Implementation details for subclasses:**
    `PsType` and its metaclass ``PsTypeMeta`` together implement a uniquing mechanism to ensure that of each type,
    only one instance ever exists in the public.
    For this to work, subclasses have to adhere to several rules:

     - All instances of `PsType` must be immutable.
     - The `const` argument must be the last keyword argument to ``__init__`` and must be passed to the superclass
       ``__init__``.
     - The `__args__` method must return a tuple of positional arguments excluding the `const` property,
       which, when passed to the class's constructor, create an identically-behaving instance.
    """

    def __init__(self, const: bool = False):
        self._const = const

        self._requalified: PsType | None = None

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
        """Arguments to this type, excluding the const-qualifier.

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
        return self._const == other._const and self.__args__() == other.__args__()

    def __str__(self) -> str:
        return self.c_string()

    def __hash__(self) -> int:
        return hash((type(self), self.__args__()))


T = TypeVar("T", bound=PsType)


def constify(t: T) -> T:
    """Adds the const qualifier to a given type."""
    if not t.const:
        if t._requalified is None:
            t._requalified = type(t)(*t.__args__(), const=True)  # type: ignore
        return cast(T, t._requalified)
    else:
        return t


def deconstify(t: T) -> T:
    """Removes the const qualifier from a given type."""
    if t.const:
        if t._requalified is None:
            t._requalified = type(t)(*t.__args__(), const=False)  # type: ignore
        return cast(T, t._requalified)
    else:
        return t
