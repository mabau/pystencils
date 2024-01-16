import sympy

import pystencils.astnodes
from pystencils.backends.cbackend import CBackend
from pystencils.typing import TypedSymbol


class BogusDeclaration(pystencils.astnodes.Node):
    """Base class for all AST nodes."""

    def __init__(self, parent=None):
        self.parent = parent

    @property
    def args(self):
        """Returns all arguments/children of this node."""
        return set()

    @property
    def symbols_defined(self):
        """Set of symbols which are defined by this node."""
        return {TypedSymbol('Foo', 'double')}

    @property
    def undefined_symbols(self):
        """Symbols which are used but are not defined inside this node."""
        set()

    def subs(self, subs_dict):
        """Inplace! substitute, similar to sympy's but modifies the AST inplace."""
        for a in self.args:
            a.subs(subs_dict)

    @property
    def func(self):
        return self.__class__

    def atoms(self, arg_type):
        """Returns a set of all descendants recursively, which are an instance of the given type."""
        result = set()
        for arg in self.args:
            if isinstance(arg, arg_type):
                result.add(arg)
            result.update(arg.atoms(arg_type))
        return result


class BogusUsage(pystencils.astnodes.Node):
    """Base class for all AST nodes."""

    def __init__(self, requires_global: bool, parent=None):
        self.parent = parent
        if requires_global:
            self.required_global_declarations = [BogusDeclaration()]

    @property
    def args(self):
        """Returns all arguments/children of this node."""
        return set()

    @property
    def symbols_defined(self):
        """Set of symbols which are defined by this node."""
        return set()

    @property
    def undefined_symbols(self):
        """Symbols which are used but are not defined inside this node."""
        return {TypedSymbol('Foo', 'double')}

    def subs(self, subs_dict):
        """Inplace! substitute, similar to sympy's but modifies the AST inplace."""
        for a in self.args:
            a.subs(subs_dict)

    @property
    def func(self):
        return self.__class__

    def atoms(self, arg_type):
        """Returns a set of all descendants recursively, which are an instance of the given type."""
        result = set()
        for arg in self.args:
            if isinstance(arg, arg_type):
                result.add(arg)
            result.update(arg.atoms(arg_type))
        return result


def test_global_definitions_with_global_symbol():
    # Teach our printer to print new ast nodes
    CBackend._print_BogusUsage = lambda _, __: "// Bogus would go here"
    CBackend._print_BogusDeclaration = lambda _, __: "// Declaration would go here"

    z, x, y = pystencils.fields("z, y, x: [2d]")

    normal_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], x[0, 0] * x[0, 0] * y[0, 0])], [])

    ast = pystencils.create_kernel(normal_assignments)
    print(pystencils.show_code(ast))
    ast.body.append(BogusUsage(requires_global=True))
    print(pystencils.show_code(ast))
    kernel = ast.compile()
    assert kernel is not None

    assert TypedSymbol('Foo', 'double') not in [p.symbol for p in ast.get_parameters()]


def test_global_definitions_without_global_symbol():
    # Teach our printer to print new ast nodes
    CBackend._print_BogusUsage = lambda _, __: "// Bogus would go here"
    CBackend._print_BogusDeclaration = lambda _, __: "// Declaration would go here"

    z, x, y = pystencils.fields("z, y, x: [2d]")

    normal_assignments = pystencils.AssignmentCollection([pystencils.Assignment(
        z[0, 0], x[0, 0] * x[0, 0] * y[0, 0])], [])

    ast = pystencils.create_kernel(normal_assignments)
    print(pystencils.show_code(ast))
    ast.body.append(BogusUsage(requires_global=False))
    print(pystencils.show_code(ast))
    kernel = ast.compile()
    assert kernel is not None

    assert TypedSymbol('Foo', 'double') in [p.symbol for p in ast.get_parameters()]
