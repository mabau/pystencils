"""
The `pystencils.types` module contains the set of classes used by pystencils
to model data types. Data types are used extensively within the code generator,
but can safely be ignored by most users unless you wish to force certain types on
symbols, generate mixed-precision kernels, et cetera.

The various classes that constitute the pystencils type system are implemented in
`pystencils.types.types`; most have abbreviated names defined in `pystencils.types.quick`.

For more information about the type system's internal workings, and developer's guidance on
how to extend it, refer to `pystencils.types.meta`.

"""

from .meta import PsType, constify, deconstify

from .types import (
    PsCustomType,
    PsStructType,
    PsNumericType,
    PsScalarType,
    PsVectorType,
    PsDereferencableType,
    PsPointerType,
    PsArrayType,
    PsBoolType,
    PsIntegerType,
    PsUnsignedIntegerType,
    PsSignedIntegerType,
    PsIeeeFloatType,
)

from .parsing import UserTypeSpec, create_type, create_numeric_type

from .exception import PsTypeError

__all__ = [
    "PsType",
    "PsCustomType",
    "PsStructType",
    "PsDereferencableType",
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
