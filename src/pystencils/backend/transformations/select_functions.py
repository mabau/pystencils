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
        return self.visit(node)

    def visit(self, node: PsAstNode) -> PsAstNode:
        node.children = [self.visit(c) for c in node.children]

        if isinstance(node, PsCall) and isinstance(node.function, PsMathFunction):
            return self._platform.select_function(node)
        else:
            return node
