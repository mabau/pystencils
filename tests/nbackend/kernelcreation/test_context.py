from itertools import chain
import pytest

from pystencils import Field, TypedSymbol, FieldType, DynamicType

from pystencils.backend.kernelcreation import KernelCreationContext
from pystencils.backend.constants import PsConstant
from pystencils.backend.exceptions import KernelConstraintsError
from pystencils.types.quick import SInt, Fp
from pystencils.types import deconstify


def test_field_arrays():
    ctx = KernelCreationContext(index_dtype=SInt(16))

    f = Field.create_generic("f", 3, Fp(32))
    f_arr = ctx.get_array(f)

    assert f_arr.element_type == f.dtype == Fp(32)
    assert len(f_arr.shape) == len(f.shape) + 1 == 4
    assert isinstance(f_arr.shape[3], PsConstant) and f_arr.shape[3].value == 1
    assert f_arr.shape[3].dtype == SInt(16, const=True)
    assert f_arr.index_type == ctx.index_dtype == SInt(16)
    assert f_arr.shape[0].dtype == ctx.index_dtype == SInt(16)

    g = Field.create_generic("g", 3, index_shape=(2, 4), dtype=Fp(16))
    g_arr = ctx.get_array(g)
    
    assert g_arr.element_type == g.dtype == Fp(16)
    assert len(g_arr.shape) == len(g.spatial_shape) + len(g.index_shape) == 5
    assert isinstance(g_arr.shape[3], PsConstant) and g_arr.shape[3].value == 2
    assert g_arr.shape[3].dtype == SInt(16, const=True)
    assert isinstance(g_arr.shape[4], PsConstant) and g_arr.shape[4].value == 4
    assert g_arr.shape[4].dtype == SInt(16, const=True)
    assert g_arr.index_type == ctx.index_dtype == SInt(16)

    h = Field(
        "h",
        FieldType.GENERIC,
        Fp(32),
        (0, 1),
        (
            TypedSymbol("nx", SInt(32)),
            TypedSymbol("ny", SInt(32)),
            1
        ),
        (
            TypedSymbol("sx", SInt(32)),
            TypedSymbol("sy", SInt(32)), 
            1
        )
        )
    
    h_arr = ctx.get_array(h)

    assert h_arr.index_type == SInt(32)
    
    for s in chain(h_arr.shape, h_arr.strides):
        assert deconstify(s.get_dtype()) == SInt(32)

    assert [s.name for s in chain(h_arr.shape[:2], h_arr.strides[:2])] == ["nx", "ny", "sx", "sy"]


def test_invalid_fields():
    ctx = KernelCreationContext(index_dtype=SInt(16))

    h = Field(
        "h",
        FieldType.GENERIC,
        Fp(32),
        (0,),
        (TypedSymbol("nx", SInt(32)),),
        (TypedSymbol("sx", SInt(64)),)
    )
    
    with pytest.raises(KernelConstraintsError):
        _ = ctx.get_array(h)

    h = Field(
        "h",
        FieldType.GENERIC,
        Fp(32),
        (0,),
        (TypedSymbol("nx", Fp(32)),),
        (TypedSymbol("sx", Fp(32)),)
    )
    
    with pytest.raises(KernelConstraintsError):
        _ = ctx.get_array(h)

    h = Field(
        "h",
        FieldType.GENERIC,
        Fp(32),
        (0,),
        (TypedSymbol("nx", DynamicType.NUMERIC_TYPE),),
        (TypedSymbol("sx", DynamicType.NUMERIC_TYPE),)
    )
    
    with pytest.raises(KernelConstraintsError):
        _ = ctx.get_array(h)
