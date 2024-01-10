#%%
import pytest
from pystencils.nbackend.types.quick import *


def test_parsing_positive():
    assert make_type("const uint32_t * restrict") == Ptr(UInt(32, const=True), restrict=True)
    assert make_type("float * * const") == Ptr(Ptr(Fp(32)), const=True)
    
def test_parsing_negative():
    bad_specs = [
        "const notatype * const",
        "cnost uint32_t",
        "int", # plain ints are ambiguous
        "float float",
        "double * int",
        "bool"
    ]

    for spec in bad_specs:
        with pytest.raises(ValueError):
            make_type(spec)


#%%
test_parsing_positive()