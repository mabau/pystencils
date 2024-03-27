from ..platforms import Platform
from ..ast import PsAstNode
from ..ast.expressions import PsCall
from ..functions import PsMathFunction


class SelectFunctions:
    """Traverse the AST to replace all instances of `PsMathFunction` by their implementation
    provided by the given `Platform`."""

    def __init__(self, platform: Platform):
        self._platform = platform

    def __call__(self, node: PsAstNode) -> PsAstNode:
        self.visit(node)
        return node

    def visit(self, node: PsAstNode):
        for c in node.children:
            self.visit(c)

        if isinstance(node, PsCall) and isinstance(node.function, PsMathFunction):
            impl = self._platform.select_function(node.function, node.get_dtype())
            node.function = impl
