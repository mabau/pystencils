import numpy as np

from .basic_types import (
    PsAbstractType,
    PsPointerType,
    PsStructType,
    PsUnsignedIntegerType,
    PsSignedIntegerType,
    PsIeeeFloatType,
)


def interpret_python_type(t: type) -> PsAbstractType:
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

    if t is np.float32:
        return PsIeeeFloatType(32)
    if t is np.float64:
        return PsIeeeFloatType(64)

    raise ValueError(f"Could not interpret Python data type {t} as a pystencils type.")


def interpret_numpy_dtype(t: np.dtype) -> PsAbstractType:
    if t.fields is not None:
        #   it's a struct
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


def parse_type_string(s: str) -> PsAbstractType:
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
                    return PsPointerType(base_type, const=False, restrict=False)
                case ["const"]:
                    return PsPointerType(base_type, const=True, restrict=False)
                case ["restrict"]:
                    return PsPointerType(base_type, const=False, restrict=True)
                case ["const", "restrict"] | ["restrict", "const"]:
                    return PsPointerType(base_type, const=True, restrict=True)
                case _:
                    raise ValueError(f"Could not parse token '{s}' as C type.")

        case _:
            raise ValueError(f"Could not parse token '{s}' as C type.")


def parse_type_name(typename: str, const: bool):
    match typename:
        case "int64" | "int64_t":
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

        case "float" | "float32":
            return PsIeeeFloatType(32, const=const)
        case "double" | "float64":
            return PsIeeeFloatType(64, const=const)

        case _:
            raise ValueError(f"Could not parse token '{typename}' as C type.")
