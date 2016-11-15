from sympy.printing.printer import Printer
from graphviz import Digraph, lang


class DotPrinter(Printer):
    """
    A printer which converts ast to DOT (graph description language).
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.dot = Digraph(**kwargs)
        self.dot.quote_edge = lang.quote

    def _print_KernelFunction(self, function):
        self.dot.node(repr(function))
        self._print(function.body)

    def _print_LoopOverCoordinate(self, loop):
        self.dot.node(repr(loop))
        self._print(loop.body)

    def _print_Block(self, block):
        for node in block.children():
            self._print(node)
        parent = block.parent
        for node in block.children():
            self.dot.edge(repr(parent), repr(node))
            parent = node

    def _print_SympyAssignment(self, assignment):
        self.dot.node(repr(assignment))

    def doprint(self, expr):
        self._print(expr)
        return self.dot.source


def dotprint(ast, view=False, **kwargs):
    """
    Returns a string which can be used to generate a DOT-graph
    :param ast: The ast which should be generated
    :param view: Boolen, if rendering of the image directly should occur.
    :param kwargs: is directly passed to the DotPrinter class: http://graphviz.readthedocs.io/en/latest/api.html#digraph
    :return: string in DOT format
    """
    printer = DotPrinter(**kwargs)
    dot = printer.doprint(ast)
    if view:
        printer.dot.render(view=view)
    return dot
