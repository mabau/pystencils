
"""
Test of pystencils.data_types.address_of
"""

from pystencils.data_types import address_of, cast_func, PointerType
import pystencils
from pystencils.simp.simplifications import sympy_cse
import sympy


def test_address_of():
    x, y = pystencils.fields('x,y: int64[2d]')
    s = pystencils.TypedSymbol('s', PointerType('int64'))

    assignments = pystencils.AssignmentCollection({
        s: address_of(x[0, 0]),
        y[0, 0]: cast_func(s, 'int64')
    }, {})

    ast = pystencils.create_kernel(assignments)
    code = pystencils.show_code(ast)
    print(code)

    assignments = pystencils.AssignmentCollection({
        y[0, 0]: cast_func(address_of(x[0, 0]), 'int64')
    }, {})

    ast = pystencils.create_kernel(assignments)
    code = pystencils.show_code(ast)
    print(code)


def test_address_of_with_cse():
    x, y = pystencils.fields('x,y: int64[2d]')
    s = pystencils.TypedSymbol('s', PointerType('int64'))

    assignments = pystencils.AssignmentCollection({
        y[0, 0]: cast_func(address_of(x[0, 0]), 'int64'),
        x[0, 0]: cast_func(address_of(x[0, 0]), 'int64') + 1
    }, {})

    ast = pystencils.create_kernel(assignments)
    code = pystencils.show_code(ast)
    assignments_cse = sympy_cse(assignments)

    ast = pystencils.create_kernel(assignments_cse)
    code = pystencils.show_code(ast)
    print(code)


def main():
    test_address_of()
    test_address_of_with_cse()


if __name__ == '__main__':
    main()
