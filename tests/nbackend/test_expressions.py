from pystencils.backend.typed_expressions import PsTypedVariable
from pystencils.backend.arrays import PsLinearizedArray, PsArrayBasePointer, PsArrayShapeVar, PsArrayStrideVar
from pystencils.backend.types.quick import *

def test_variable_equality():
    var1 = PsTypedVariable("x", Fp(32))
    var2 = PsTypedVariable("x", Fp(32))
    assert var1 == var2

    shape = (..., ..., ...)
    strides = (..., ..., ...)

    arr = PsLinearizedArray("arr", Fp(64), shape, strides)
    bp1 = PsArrayBasePointer("arr_data", arr)
    bp2 = PsArrayBasePointer("arr_data", arr)
    assert bp1 == bp2

    arr1 = PsLinearizedArray("arr", Fp(64), shape, strides)
    bp1 = PsArrayBasePointer("arr_data", arr1)

    arr2 = PsLinearizedArray("arr", Fp(64), shape, strides)
    bp2 = PsArrayBasePointer("arr_data", arr2)
    assert bp1 == bp2

    for v1, v2 in zip(arr1.shape, arr2.shape):
        assert v1 == v2

    for v1, v2 in zip(arr1.strides, arr2.strides):
        assert v1 == v2


def test_variable_inequality():
    shape = (..., ..., ...)
    strides = (..., ..., ...)

    var1 = PsTypedVariable("x", Fp(32))
    var2 = PsTypedVariable("x", Fp(64))
    assert var1 != var2

    var1 = PsTypedVariable("x", Fp(32, True))
    var2 = PsTypedVariable("x", Fp(32, False))
    assert var1 != var2

    #   Arrays 
    arr1 = PsLinearizedArray("arr", Fp(64), shape, strides)
    bp1 = PsArrayBasePointer("arr_data", arr1)

    arr2 = PsLinearizedArray("arr", Fp(32), shape, strides)
    bp2 = PsArrayBasePointer("arr_data", arr2)
    assert bp1 != bp2

