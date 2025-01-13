from __future__ import annotations

from warnings import warn
from typing import Sequence, Iterable

from .properties import (
    PsSymbolProperty,
    _FieldProperty,
    FieldShape,
    FieldStride,
    FieldBasePtr,
)
from ..types import PsType
from ..field import Field
from ..sympyextensions import TypedSymbol


class Parameter:
    """Parameter to an output object of the code generator."""

    __match_args__ = ("name", "dtype", "properties")

    def __init__(
        self, name: str, dtype: PsType, properties: Iterable[PsSymbolProperty] = ()
    ):
        self._name = name
        self._dtype = dtype
        self._properties: frozenset[PsSymbolProperty] = (
            frozenset(properties) if properties is not None else frozenset()
        )
        self._fields: tuple[Field, ...] = tuple(
            sorted(
                set(
                    p.field  # type: ignore
                    for p in filter(
                        lambda p: isinstance(p, _FieldProperty), self._properties
                    )
                ),
                key=lambda f: f.name,
            )
        )

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    def _hashable_contents(self):
        return (self._name, self._dtype, self._properties)

    #   TODO: Need?
    def __hash__(self) -> int:
        return hash(self._hashable_contents())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Parameter):
            return False

        return (
            type(self) is type(other)
            and self._hashable_contents() == other._hashable_contents()
        )

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name = {self._name}, dtype = {self._dtype})"

    @property
    def symbol(self) -> TypedSymbol:
        return TypedSymbol(self.name, self.dtype)

    @property
    def fields(self) -> Sequence[Field]:
        """Set of fields associated with this parameter."""
        return self._fields

    def get_properties(
        self, prop_type: type[PsSymbolProperty] | tuple[type[PsSymbolProperty], ...]
    ) -> set[PsSymbolProperty]:
        """Retrieve all properties of the given type(s) attached to this parameter"""
        return set(filter(lambda p: isinstance(p, prop_type), self._properties))

    @property
    def properties(self) -> frozenset[PsSymbolProperty]:
        return self._properties

    @property
    def is_field_parameter(self) -> bool:
        return bool(self._fields)

    #   Deprecated legacy properties
    #   These are kept mostly for the legacy waLBerla code generation system

    @property
    def is_field_pointer(self) -> bool:  # pragma: no cover
        warn(
            "`is_field_pointer` is deprecated and will be removed in a future version of pystencils. "
            "Use `param.get_properties(FieldBasePtr)` instead.",
            DeprecationWarning,
        )
        return bool(self.get_properties(FieldBasePtr))

    @property
    def is_field_stride(self) -> bool:  # pragma: no cover
        warn(
            "`is_field_stride` is deprecated and will be removed in a future version of pystencils. "
            "Use `param.get_properties(FieldStride)` instead.",
            DeprecationWarning,
        )
        return bool(self.get_properties(FieldStride))

    @property
    def is_field_shape(self) -> bool:  # pragma: no cover
        warn(
            "`is_field_shape` is deprecated and will be removed in a future version of pystencils. "
            "Use `param.get_properties(FieldShape)` instead.",
            DeprecationWarning,
        )
        return bool(self.get_properties(FieldShape))

    @property
    def field_name(self) -> str:  # pragma: no cover
        warn(
            "`field_name` is deprecated and will be removed in a future version of pystencils. "
            "Use `param.fields[0].name` instead.",
            DeprecationWarning,
        )
        return self._fields[0].name
