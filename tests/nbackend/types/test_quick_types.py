import pytest
from pystencils.nbackend.types.quick import *


def test_parsing_positive():
    assert make_type("const uint32_t * restrict") == Ptr(UInt(32, const=True), restrict=True)
    assert make_type("float * * const") == Ptr(Ptr(Fp(32)), const=True)
    assert make_type("uint16 * const") == Ptr(UInt(16), const=True)
    assert make_type("uint64 const * const") == Ptr(UInt(64, const=True), const=True)
    
def test_parsing_negative():
    bad_specs = [
        "const notatype * const",
        "cnost uint32_t",
        "uint45_t",
        "int", # plain ints are ambiguous
        "float float",
        "double * int",
        "bool"
    ]

    for spec in bad_specs:
        with pytest.raises(ValueError):
            make_type(spec)

def test_numpy():
    import numpy as np
    assert make_type(np.single) == make_type(np.float32) == PsIeeeFloatType(32)
    assert make_type(float) == make_type(np.double) == make_type(np.float64) == PsIeeeFloatType(64)
    assert make_type(int) == make_type(np.int64) == PsSignedIntegerType(64)
