"""Abbreviations and creation functions for pystencils type-modelling classes
for quick, user-friendly construction and compact pattern matching.

This module is meant to be included whole, e.g. as `from pystencils.nbackend.types.quick import *`
"""

from __future__ import annotations

import numpy as np

from .basic_types import (
    PsType,
    PsCustomType,
    PsNumericType,
    PsScalarType,
    PsBoolType,
    PsPointerType,
    PsIntegerType,
    PsUnsignedIntegerType,
    PsSignedIntegerType,
    PsIeeeFloatType,
)

UserTypeSpec = str | type | np.dtype | PsType


def create_type(type_spec: UserTypeSpec) -> PsType:
    """Create a pystencils type object from a variety of specifications.

    Possible arguments are:
        - Strings ('str'): will be parsed as common C types, throwing an exception if that fails.
          To construct a `PsCustomType` instead, use the constructor of `PsCustomType` or its abbreviation
          ``types.quick.Custom`` instead
        - Python builtin data types (instances of `type`): Attempts to interpret Python numeric types like so:
            - `int` becomes a signed 64-bit integer
            - `float` becomes a double-precision IEEE-754 float
            - No others are supported at the moment
        - Supported Numpy scalar data types (see https://numpy.org/doc/stable/reference/arrays.scalars.html)
          are converted to pystencils scalar data types
        - Instances of `numpy.dtype`: Attempt to interpret scalar types like above, and structured types as structs.
        - Instances of `PsAbstractType` will be returned as they are
    """

    from .parsing import parse_type_string, interpret_python_type, interpret_numpy_dtype

    if isinstance(type_spec, PsType):
        return type_spec
    if isinstance(type_spec, str):
        return parse_type_string(type_spec)
    if isinstance(type_spec, type):
        return interpret_python_type(type_spec)
    if isinstance(type_spec, np.dtype):
        return interpret_numpy_dtype(type_spec)
    raise ValueError(f"{type_spec} is not a valid type specification.")


def create_numeric_type(type_spec: UserTypeSpec) -> PsNumericType:
    """Like `make_type`, but only for numeric types."""
    dtype = create_type(type_spec)
    if not isinstance(dtype, PsNumericType):
        raise ValueError(
            f"Given type {type_spec} does not translate to a numeric type."
        )
    return dtype


Custom = PsCustomType
"""`Custom(name)` matches `PsCustomType(name)`"""

Scalar = PsScalarType
"""`Scalar()` matches any subclass of `PsScalarType`"""

Ptr = PsPointerType
"""`Ptr(t)` matches `PsPointerType(base_type=t)`"""

Bool = PsBoolType
"""Bool() matches PsBoolType()"""

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
