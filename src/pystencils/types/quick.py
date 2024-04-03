"""Quick access to the pystencils data type system."""

from __future__ import annotations

from .types import (
    PsCustomType,
    PsScalarType,
    PsBoolType,
    PsPointerType,
    PsArrayType,
    PsIntegerType,
    PsUnsignedIntegerType,
    PsSignedIntegerType,
    PsIeeeFloatType,
)

Custom = PsCustomType
"""Custom data types are modelled only by their name."""

Scalar = PsScalarType
"""``Scalar()`` matches any subclass of ``PsScalarType``"""

Ptr = PsPointerType
"""``Ptr(t)`` matches ``PsPointerType(base_type=t)``"""

Arr = PsArrayType
"""``Arr(t, s)`` matches ``PsArrayType(base_type=t, size=s)``"""

Bool = PsBoolType
"""``Bool()`` matches ``PsBoolType()``"""

AnyInt = PsIntegerType
"""``AnyInt(width)`` matches both ``PsUnsignedIntegerType(width)`` and ``PsSignedIntegerType(width)``"""

UInt = PsUnsignedIntegerType
"""``UInt(width)`` matches ``PsUnsignedIntegerType(width)``"""

Int = PsSignedIntegerType
"""``Int(width)`` matches ``PsSignedIntegerType(width)``"""

SInt = PsSignedIntegerType
"""``SInt(width)`` matches ``PsSignedIntegerType(width)``"""

Fp = PsIeeeFloatType
"""``Fp(width)`` matches ``PsIeeeFloatType(width)``"""
