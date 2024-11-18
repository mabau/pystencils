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
"""Alias of `PsCustomType`"""

Scalar = PsScalarType
"""Alias of `PsScalarType`"""

Ptr = PsPointerType
"""Alias of `PsPointerType`"""

Arr = PsArrayType
"""Alias of `PsArrayType`"""

Bool = PsBoolType
"""Alias of `PsBoolType`"""

AnyInt = PsIntegerType
"""Alias of `PsIntegerType`"""

UInt = PsUnsignedIntegerType
"""Alias of `PsUnsignedIntegerType`"""

Int = PsSignedIntegerType
"""Alias of `PsSignedIntegerType`"""

SInt = PsSignedIntegerType
"""Alias of `PsSignedIntegerType`"""

Fp = PsIeeeFloatType
"""Alias of `PsIeeeFloatType`"""
