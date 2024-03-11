"""
The `pystencils.types` module contains the set of classes used by pystencils
to model data types. Data types are used extensively within the code generator,
but can safely be ignored by most users unless you wish to force certain types on
symbols, generate mixed-precision kernels, et cetera.

For more user-friendly and less verbose access to the type modelling system, refer to
the `pystencils.types.quick` submodule. 
"""

from .basic_types import (
    PsType,
    PsCustomType,
    PsStructType,
    PsNumericType,
    PsScalarType,
    PsVectorType,
    PsSubscriptableType,
    PsPointerType,
    PsArrayType,
    PsBoolType,
    PsIntegerType,
    PsUnsignedIntegerType,
    PsSignedIntegerType,
    PsIeeeFloatType,
    constify,
    deconstify,
)

from .quick import UserTypeSpec, create_type, create_numeric_type

from .exception import PsTypeError

__all__ = [
    "PsType",
    "PsCustomType",
    "PsStructType",
    "PsSubscriptableType",
    "PsPointerType",
    "PsArrayType",
    "PsNumericType",
    "PsScalarType",
    "PsVectorType",
    "PsIntegerType",
    "PsBoolType",
    "PsUnsignedIntegerType",
    "PsSignedIntegerType",
    "PsIeeeFloatType",
    "constify",
    "deconstify",
    "UserTypeSpec",
    "create_type",
    "create_numeric_type",
    "PsTypeError",
]
