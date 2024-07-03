"""
Test of pystencils.data_types.address_of
"""
import pytest
import pystencils
from pystencils.types import PsPointerType, create_type
from pystencils.sympyextensions.pointers import AddressOf
from pystencils.sympyextensions.typed_sympy import CastFunc
from pystencils.sympyextensions import sympy_cse

import sympy as sp


def test_address_of():
    x, y = pystencils.fields('x, y: int64[2d]')
    s = pystencils.TypedSymbol('s', PsPointerType(create_type('int64')))

    assert AddressOf(x[0, 0]).canonical() == x[0, 0]
    assert AddressOf(x[0, 0]).dtype == PsPointerType(x[0, 0].dtype, restrict=True, const=True)
    
    with pytest.raises(ValueError):
        assert AddressOf(sp.Symbol("a")).dtype

    assignments = pystencils.AssignmentCollection({
        s: AddressOf(x[0, 0]),
        y[0, 0]: CastFunc(s, create_type('int64'))
    })

    _ = pystencils.create_kernel(assignments).compile()
    # pystencils.show_code(kernel.ast)

    assignments = pystencils.AssignmentCollection({
        y[0, 0]: CastFunc(AddressOf(x[0, 0]), create_type('int64'))
    })

    _ = pystencils.create_kernel(assignments).compile()
    # pystencils.show_code(kernel.ast)


def test_address_of_with_cse():
    x, y = pystencils.fields('x, y: int64[2d]')

    assignments = pystencils.AssignmentCollection({
        x[0, 0]: CastFunc(AddressOf(x[0, 0]), create_type('int64')) + 1
    })

    _ = pystencils.create_kernel(assignments).compile()
    # pystencils.show_code(kernel.ast)
    assignments_cse = sympy_cse(assignments)

    _ = pystencils.create_kernel(assignments_cse).compile()
    # pystencils.show_code(kernel.ast)
