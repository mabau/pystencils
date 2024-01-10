"""Abbreviations and creation functions for pystencils type-modelling classes
for quick, user-friendly construction and compact pattern matching.

This module is meant to be included whole, e.g. as `from pystencils.nbackend.types.quick import *`
"""

from __future__ import annotations

from .basic_types import (
    PsAbstractType,
    PsCustomType,
    PsScalarType,
    PsPointerType,
    PsIntegerType,
    PsUnsignedIntegerType,
    PsSignedIntegerType,
    PsIeeeFloatType,
)

UserTypeSpec = str | type | PsAbstractType


def make_type(type_spec: UserTypeSpec) -> PsAbstractType:
    """Create a pystencils type object from a variety of specifications.

    Possible arguments are:
        - Strings ('str'): will be parsed as common C types, throwing an exception if that fails.
          To construct a `PsCustomType` instead, use the constructor of `PsCustomType` or its abbreviation
          `types.quick.Custom` instead
        - Python builtin data types (instances of `type`): Attempts to interpret Python numeric types like so:
            - `int` becomes a signed 64-bit integer
            - `float` becomes a double-precision IEEE-754 float
            - No others are supported at the moment
        - Supported Numpy scalar data types (see https://numpy.org/doc/stable/reference/arrays.scalars.html) are converted to pystencils
          scalar data types
        - Instances of `PsAbstractType` will be returned as they are
    """

    from .parsing import (
        parse_type_string,
        interpret_python_type,
    )

    if isinstance(type_spec, PsAbstractType):
        return type_spec
    if isinstance(type_spec, str):
        return parse_type_string(type_spec)
    if isinstance(type_spec, type):
        return interpret_python_type(type_spec)
    raise ValueError(f"{type_spec} is not a valid type specification.")


Custom = PsCustomType
"""`Custom(name)` matches `PsCustomType(name)`"""

Scalar = PsScalarType
"""`Scalar()` matches any subclass of `PsScalarType`"""

Ptr = PsPointerType
"""`Ptr(t)` matches `PsPointerType(base_type=t)`"""

AnyInt = PsIntegerType
"""`AnyInt(width)` matches both `PsUnsignedIntegerType(width)` and `PsSignedIntegerType(width)`"""

UInt = PsUnsignedIntegerType
"""`UInt(width)` matches `PsUnsignedIntegerType(width)`"""

Int = PsSignedIntegerType
"""`Int(width)` matches `PsSignedIntegerType(width)`"""

SInt = PsSignedIntegerType
"""`SInt(width)` matches `PsSignedIntegerType(width)`"""

Fp = PsIeeeFloatType
"""`Fp(width)` matches `PsIeeeFloatType(width)`"""
