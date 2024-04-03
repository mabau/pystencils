"""
Although mostly invisible to the user, types are ubiquitous throughout pystencils.
They are created and converted in many places, especially in the code generation backend.
To handle and compare types more efficiently, the pystencils type system implements
a uniquing mechanism to ensure that at any point there exists only one instance of each type.
This means, for example, if a 32-bit const unsigned integer type gets created in two places
at two different times in the program, the two types don't just behave identically, but
in fact refer to the same object:

>>> from pystencils.types import PsUnsignedIntegerType
>>> t1 = PsUnsignedIntegerType(32, const=True)
>>> t2 = PsUnsignedIntegerType(32, const=True)
>>> t1 is t2
True

Both calls to `PsUnsignedIntegerType` return the same object. This is ensured by the
`PsTypeMeta` metaclass.
This metaclass holds an internal registry of all type objects ever created,
and alters the class instantiation mechanism such that whenever a type is instantiated
a second time with the same arguments, the pre-existing instance is found and returned instead.

For this to work, all instantiable subclasses of `PsType` must implement the following protocol:

- The ``const`` parameter must be the last keyword parameter of ``__init__``.
- The ``__canonical_args__`` classmethod must have the same signature as ``__init__``, except it does
  not take the ``const`` parameter. It must return a tuple containing all the positional and keyword
  arguments in their canonical order. This method is used by `PsTypeMeta` to identify instances of the type,
  and to catch the various different possibilities Python offers for passing function arguments.
- The ``__args__`` method, when called on an instance of the type, must return a tuple containing the constructor
  arguments required to create that exact instance.

Developers intending to extend the type class hierarchy are advised to study the implementations
of this protocol in the existing classes.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TypeVar, Any, cast
import numpy as np


class PsTypeMeta(ABCMeta):
    """Metaclass for the `PsType` hierarchy.

    `PsTypeMeta` holds an internal cache of all instances of `PsType` and overrides object creation
    such that whenever a type gets instantiated more than once, instead of creating a new object,
    the existing object is returned.
    """

    _instances: dict[Any, PsType] = dict()

    def __call__(
        cls: PsTypeMeta, *args: Any, const: bool = False, **kwargs: Any
    ) -> Any:
        assert issubclass(cls, PsType)
        canonical_args = cls.__canonical_args__(*args, **kwargs)
        key = (cls, canonical_args, const)

        if key in cls._instances:
            obj = cls._instances[key]
        else:
            obj = super().__call__(*args, const=const, **kwargs)
            cls._instances[key] = obj

        return obj


class PsType(metaclass=PsTypeMeta):
    """Base class for all pystencils types.

    Args:
        const: Const-qualification of this type
    """

    def __new__(cls, *args, _pickle=False, **kwargs):
        if _pickle:
            #   force unpickler to use metaclass uniquing mechanism
            return cls(*args, **kwargs)
        else:
            return super().__new__(cls)

    def __getnewargs_ex__(self):
        args = self.__args__()
        kwargs = {"const": self._const, "_pickle": True}
        return args, kwargs

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
        For each instantiable subclass ``MyType`` of ``PsType``, the following must hold::

            t = MyType(< arguments >)
            assert MyType(*t.__args__()) == t

        """
        pass

    @classmethod
    @abstractmethod
    def __canonical_args__(cls, *args, **kwargs):
        """Return a tuple containing the positional and keyword arguments of ``__init__``
        in their canonical order."""
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
