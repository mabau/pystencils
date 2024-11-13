"""

Caching of Instances
^^^^^^^^^^^^^^^^^^^^

To handle and compare types more efficiently, the pystencils type system customizes class
instantiation to cache and reuse existing instances of types.
This means, for example, if a 32-bit const unsigned integer type gets created in two places
in the program, the resulting objects are exactly the same:

>>> from pystencils.types import PsUnsignedIntegerType
>>> t1 = PsUnsignedIntegerType(32, const=True)
>>> t2 = PsUnsignedIntegerType(32, const=True)
>>> t1 is t2
True

This mechanism is implemented by the metaclass `PsTypeMeta`. It is not perfect, however;
some parts of Python that bypass the regular object creation sequence, such as `pickle` and
`copy.copy`, may create additional instances of types.

.. autoclass:: pystencils.types.meta.PsTypeMeta
    :members:

Extending the Type System
^^^^^^^^^^^^^^^^^^^^^^^^^

When extending the type system's class hierarchy, new classes need to implement at least the internal
method `__args__`. This method, when called on a type object, must return a hashable sequence of arguments
-- not including the const-qualifier --
that can be used to recreate that exact type. It is used internally to compute hashes and compare equality
of types, as well as for const-conversion.
    
.. autofunction:: pystencils.types.PsType.__args__

"""

from __future__ import annotations

from warnings import warn
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Any, cast
import numpy as np


class PsTypeMeta(ABCMeta):
    """Metaclass for the `PsType` hierarchy.

    `PsTypeMeta` holds an internal cache of all created instances of `PsType` and overrides object creation
    such that whenever a type gets instantiated more than once with the same argument list,
    instead of creating a new object, the existing object is returned.
    """

    _instances: dict[Any, PsType] = dict()

    def __call__(cls: PsTypeMeta, *args: Any, **kwargs: Any) -> Any:
        assert issubclass(cls, PsType)
        kwarg_tuples = tuple(sorted(kwargs.items(), key=lambda t: t[0]))

        try:
            key = (cls, args, kwarg_tuples)

            if key in cls._instances:
                return cls._instances[key]
        except TypeError:
            key = None

        obj = super().__call__(*args, **kwargs)
        canonical_key = (cls, obj.__args__(), (("const", obj.const),))

        if canonical_key in cls._instances:
            obj = cls._instances[canonical_key]
        else:
            cls._instances[canonical_key] = obj

        if key is not None:
            cls._instances[key] = obj

        return obj


class PsType(metaclass=PsTypeMeta):
    """Base class for all pystencils types.

    Args:
        const: Const-qualification of this type
    """

    #   -------------------------------------------------------------------------------------------
    #   Arguments, Equality and Hashing
    #   -------------------------------------------------------------------------------------------

    @abstractmethod
    def __args__(self) -> tuple[Any, ...]:
        """Return the arguments used to create this instance, in canonical order, excluding the const-qualifier.

        The tuple returned by this method must be hashable and for each instantiable subclass
        ``MyType`` of ``PsType``, the following must hold::

            t = MyType(< arguments >)
            assert MyType(*t.__args__(), const=t.const) == t

        """

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True

        if type(self) is not type(other):
            return False

        other = cast(PsType, other)
        return self.const == other.const and self.__args__() == other.__args__()

    def __hash__(self) -> int:
        return hash((type(self), self.const, self.__args__()))

    #   -------------------------------------------------------------------------------------------
    #   Constructor and properties
    #   -------------------------------------------------------------------------------------------

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
        """If this type has a valid in-memory size, return that size in bytes."""
        return None

    @property
    def numpy_dtype(self) -> np.dtype | None:
        """A np.dtype object representing this data type.

        Available both for backward compatibility and for interaction with the numpy-based runtime system.
        """
        return None

    #   -------------------------------------------------------------------------------------------
    #   String Conversion
    #   -------------------------------------------------------------------------------------------

    def _const_string(self) -> str:
        return "const " if self._const else ""

    @abstractmethod
    def c_string(self) -> str:
        pass

    @property
    def c_name(self) -> str:
        """Returns the C name of this type without const-qualifiers."""
        warn(
            "`c_name` is deprecated and will be removed in a future version of pystencils. "
            "Use `c_string()` instead.",
            DeprecationWarning,
        )
        return deconstify(self).c_string()

    def __str__(self) -> str:
        return self.c_string()


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
