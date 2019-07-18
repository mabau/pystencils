import sympy
import jinja2


import pystencils
from pystencils.astnodes import DestructuringBindingsForFieldClass
from pystencils.kernelparameters import  FieldPointerSymbol, FieldShapeSymbol, FieldStrideSymbol



def test_destructuring_field_class():
    z, x, y = pystencils.fields("z, y, x: [2d]")

    normal_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], x[0, 0] * sympy.log(x[0, 0] * y[0, 0]))], [])

    ast = pystencils.create_kernel(normal_assignments, target='gpu')
    print(pystencils.show_code(ast))

    ast.body = DestructuringBindingsForFieldClass(ast.body)
    print(pystencils.show_code(ast))
    ast.compile()


class DestructuringEmojiClass(DestructuringBindingsForFieldClass):
    CLASS_TO_MEMBER_DICT = {
        FieldPointerSymbol: "🥶",
        FieldShapeSymbol: "😳_%i",
        FieldStrideSymbol: "🥵_%i"
    }
    CLASS_NAME_TEMPLATE = jinja2.Template("🤯<{{ dtype }}, {{ ndim }}>")
    def __init__(self, node):
        super().__init__(node)
        self.headers = []
        
    
def test_destructuring_alternative_field_class():
    z, x, y = pystencils.fields("z, y, x: [2d]")

    normal_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], x[0, 0] * sympy.log(x[0, 0] * y[0, 0]))], [])

    ast = pystencils.create_kernel(normal_assignments, target='gpu')
    ast.body = DestructuringEmojiClass(ast.body)
    print(pystencils.show_code(ast))

def main():
    test_destructuring_field_class()
    test_destructuring_alternative_field_class()


if __name__ == '__main__':
    main()
