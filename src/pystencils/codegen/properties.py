from __future__ import annotations
from dataclasses import dataclass

from ..field import Field


@dataclass(frozen=True)
class PsSymbolProperty:
    """Base class for symbol properties, which can be used to add additional information to symbols"""


@dataclass(frozen=True)
class UniqueSymbolProperty(PsSymbolProperty):
    """Base class for unique properties, of which only one instance may be registered at a time."""


@dataclass(frozen=True)
class FieldShape(PsSymbolProperty):
    """Symbol acts as a shape parameter to a field."""

    field: Field
    coordinate: int


@dataclass(frozen=True)
class FieldStride(PsSymbolProperty):
    """Symbol acts as a stride parameter to a field."""

    field: Field
    coordinate: int


@dataclass(frozen=True)
class FieldBasePtr(UniqueSymbolProperty):
    """Symbol acts as a base pointer to a field."""

    field: Field


FieldProperty = FieldShape | FieldStride | FieldBasePtr
_FieldProperty = (FieldShape, FieldStride, FieldBasePtr)
