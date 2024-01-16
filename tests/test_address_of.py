"""
Test of pystencils.data_types.address_of
"""
import pytest
import pystencils
from pystencils.typing import PointerType, CastFunc, BasicType
from pystencils.functions import AddressOf
from pystencils.simp.simplifications import sympy_cse

import sympy as sp


def test_address_of():
    x, y = pystencils.fields('x, y: int64[2d]')
    s = pystencils.TypedSymbol('s', PointerType(BasicType('int64')))

    assert AddressOf(x[0, 0]).canonical() == x[0, 0]
    assert AddressOf(x[0, 0]).dtype == PointerType(x[0, 0].dtype, restrict=True)
    with pytest.raises(ValueError):
        assert AddressOf(sp.Symbol("a")).dtype

    assignments = pystencils.AssignmentCollection({
        s: AddressOf(x[0, 0]),
        y[0, 0]: CastFunc(s, BasicType('int64'))
    })

    kernel = pystencils.create_kernel(assignments).compile()
    # pystencils.show_code(kernel.ast)

    assignments = pystencils.AssignmentCollection({
        y[0, 0]: CastFunc(AddressOf(x[0, 0]), BasicType('int64'))
    })

    kernel = pystencils.create_kernel(assignments).compile()
    # pystencils.show_code(kernel.ast)


def test_address_of_with_cse():
    x, y = pystencils.fields('x, y: int64[2d]')

    assignments = pystencils.AssignmentCollection({
        x[0, 0]: CastFunc(AddressOf(x[0, 0]), BasicType('int64')) + 1
    })

    kernel = pystencils.create_kernel(assignments).compile()
    # pystencils.show_code(kernel.ast)
    assignments_cse = sympy_cse(assignments)

    kernel = pystencils.create_kernel(assignments_cse).compile()
    # pystencils.show_code(kernel.ast)
