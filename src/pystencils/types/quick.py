"""Quick access to the pystencils data type system."""

from __future__ import annotations

import numpy as np

from .basic_types import (
    PsType,
    PsCustomType,
    PsNumericType,
    PsScalarType,
    PsBoolType,
    PsPointerType,
    PsArrayType,
    PsIntegerType,
    PsUnsignedIntegerType,
    PsSignedIntegerType,
    PsIeeeFloatType,
)

UserTypeSpec = str | type | np.dtype | PsType


def create_type(type_spec: UserTypeSpec) -> PsType:
    """Create a pystencils type object from a variety of specifications.

    This function converts several possible representations of data types to an instance of `PsType`.
    The ``type_spec`` argument can be any of the following:

    - Strings (`str`): will be parsed as common C types, throwing an exception if that fails.
      To construct a `PsCustomType` instead, use the constructor of `PsCustomType`
      or its abbreviation `types.quick.Custom`.
    - Python builtin data types (instances of `type`): Attempts to interpret Python numeric types like so:
        - `int` becomes a signed 64-bit integer
        - `float` becomes a double-precision IEEE-754 float
        - No others are supported at the moment
    - Supported Numpy scalar data types (see https://numpy.org/doc/stable/reference/arrays.scalars.html)
      are converted to pystencils scalar data types
    - Instances of `numpy.dtype`: Attempt to interpret scalar types like above, and structured types as structs.
    - Instances of `PsType` will be returned as they are

    Args:
        type_spec: The data type, in one of the above formats
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
    """Like `create_type`, but only for numeric types."""
    dtype = create_type(type_spec)
    if not isinstance(dtype, PsNumericType):
        raise ValueError(
            f"Given type {type_spec} does not translate to a numeric type."
        )
    return dtype


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
"""``SInt(width)` matches `PsSignedIntegerType(width)``"""

Fp = PsIeeeFloatType
"""``Fp(width)` matches `PsIeeeFloatType(width)``"""
