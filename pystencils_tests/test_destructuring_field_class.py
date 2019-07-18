import sympy

import pystencils
from pystencils.astnodes import DestructuringBindingsForFieldClass


def test_destructuring_field_class():
    z, x, y = pystencils.fields("z, y, x: [2d]")

    normal_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], x[0, 0] * sympy.log(x[0, 0] * y[0, 0]))], [])

    ast = pystencils.create_kernel(normal_assignments)
    print(pystencils.show_code(ast))

    ast.body = DestructuringBindingsForFieldClass(ast.body)
    print(pystencils.show_code(ast))


def main():
    test_destructuring_field_class()


if __name__ == '__main__':
    main()
