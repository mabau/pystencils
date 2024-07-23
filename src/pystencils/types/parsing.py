import numpy as np

from .types import (
    PsType,
    PsPointerType,
    PsStructType,
    PsNumericType,
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


def interpret_python_type(t: type) -> PsType:
    if t is int:
        return PsSignedIntegerType(64)
    if t is float:
        return PsIeeeFloatType(64)

    if t is np.uint8:
        return PsUnsignedIntegerType(8)
    if t is np.uint16:
        return PsUnsignedIntegerType(16)
    if t is np.uint32:
        return PsUnsignedIntegerType(32)
    if t is np.uint64:
        return PsUnsignedIntegerType(64)

    if t is np.int8:
        return PsSignedIntegerType(8)
    if t is np.int16:
        return PsSignedIntegerType(16)
    if t is np.int32:
        return PsSignedIntegerType(32)
    if t is np.int64:
        return PsSignedIntegerType(64)

    if t is np.float16:
        return PsIeeeFloatType(16)
    if t is np.float32:
        return PsIeeeFloatType(32)
    if t is np.float64:
        return PsIeeeFloatType(64)

    raise ValueError(f"Could not interpret Python data type {t} as a pystencils type.")


def interpret_numpy_dtype(t: np.dtype) -> PsType:
    if t.fields is not None:
        #   it's a struct
        if not t.isalignedstruct:
            raise ValueError("pystencils currently only accepts aligned structured data types.")

        members = []
        for fname, fspec in t.fields.items():
            members.append(PsStructType.Member(fname, interpret_numpy_dtype(fspec[0])))
        return PsStructType(members)
    else:
        try:
            return interpret_python_type(t.type)
        except ValueError:
            raise ValueError(
                f"Could not interpret numpy dtype object {t} as a pystencils type."
            )


def parse_type_string(s: str) -> PsType:
    tokens = s.rsplit("*", 1)
    match tokens:
        case [base]:  # input contained no '*', is no pointer
            match base.split():  # split at whitespace to find `const` qualifiers (C typenames cannot contain spaces)
                case [typename]:
                    return parse_type_name(typename, False)
                case ["const", typename] | [typename, "const"]:
                    return parse_type_name(typename, True)
                case _:
                    raise ValueError(f"Could not parse token '{base}' as C type.")

        case [base, suffix]:  # input was "base * suffix"
            base_type = parse_type_string(base)
            match suffix.split():
                case []:
                    return PsPointerType(base_type, restrict=False, const=False)
                case ["const"]:
                    return PsPointerType(base_type, restrict=False, const=True)
                case ["restrict"]:
                    return PsPointerType(base_type, restrict=True, const=False)
                case ["const", "restrict"] | ["restrict", "const"]:
                    return PsPointerType(base_type, restrict=True, const=True)
                case _:
                    raise ValueError(f"Could not parse token '{s}' as C type.")

        case _:
            raise ValueError(f"Could not parse token '{s}' as C type.")


def parse_type_name(typename: str, const: bool):
    match typename:
        case "int" | "int64" | "int64_t":
            return PsSignedIntegerType(64, const=const)
        case "int32" | "int32_t":
            return PsSignedIntegerType(32, const=const)
        case "int16" | "int16_t":
            return PsSignedIntegerType(16, const=const)
        case "int8" | "int8_t":
            return PsSignedIntegerType(8, const=const)

        case "uint64" | "uint64_t":
            return PsUnsignedIntegerType(64, const=const)
        case "uint32" | "uint32_t":
            return PsUnsignedIntegerType(32, const=const)
        case "uint16" | "uint16_t":
            return PsUnsignedIntegerType(16, const=const)
        case "uint8" | "uint8_t":
            return PsUnsignedIntegerType(8, const=const)

        case "half" | "float16":
            return PsIeeeFloatType(16, const=const)
        case "float" | "float32":
            return PsIeeeFloatType(32, const=const)
        case "double" | "float64":
            return PsIeeeFloatType(64, const=const)

        case _:
            raise ValueError(f"Could not parse token '{typename}' as C type.")
