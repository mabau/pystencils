"""
Test of pystencils.data_types.address_of
"""
import sympy as sp
import pystencils
from pystencils.data_types import PointerType, address_of, cast_func, create_type
from pystencils.simp.simplifications import sympy_cse


def test_address_of():
    x, y = pystencils.fields('x,y: int64[2d]')
    s = pystencils.TypedSymbol('s', PointerType(create_type('int64')))

    assert address_of(x[0, 0]).canonical() == x[0, 0]
    assert address_of(x[0, 0]).dtype == PointerType(x[0, 0].dtype, restrict=True)
    assert address_of(sp.Symbol("a")).dtype == PointerType('void', restrict=True)

    assignments = pystencils.AssignmentCollection({
        s: address_of(x[0, 0]),
        y[0, 0]: cast_func(s, create_type('int64'))
    }, {})

    ast = pystencils.create_kernel(assignments)
    pystencils.show_code(ast)

    assignments = pystencils.AssignmentCollection({
        y[0, 0]: cast_func(address_of(x[0, 0]), create_type('int64'))
    }, {})

    ast = pystencils.create_kernel(assignments)
    pystencils.show_code(ast)


def test_address_of_with_cse():
    x, y = pystencils.fields('x,y: int64[2d]')
    s = pystencils.TypedSymbol('s', PointerType(create_type('int64')))

    assignments = pystencils.AssignmentCollection({
        y[0, 0]: cast_func(address_of(x[0, 0]), create_type('int64')) + s,
        x[0, 0]: cast_func(address_of(x[0, 0]), create_type('int64')) + 1
    }, {})

    ast = pystencils.create_kernel(assignments)
    pystencils.show_code(ast)
    assignments_cse = sympy_cse(assignments)

    ast = pystencils.create_kernel(assignments_cse)
    pystencils.show_code(ast)
