from sympy.printing.printer import Printer
from graphviz import Digraph, lang


class DotPrinter(Printer):
    """
    A printer which converts ast to DOT (graph description language).
    """
    def __init__(self, nodeToStrFunction, full, **kwargs):
        super(DotPrinter, self).__init__()
        self._nodeToStrFunction = nodeToStrFunction
        self.full = full
        self.dot = Digraph(**kwargs)
        self.dot.quote_edge = lang.quote

    def _print_KernelFunction(self, function):
        print(self._nodeToStrFunction(function))
        self.dot.node(self._nodeToStrFunction(function), style='filled', fillcolor='#E69F00')
        self._print(function.body)

    def _print_LoopOverCoordinate(self, loop):
        self.dot.node(self._nodeToStrFunction(loop), style='filled', fillcolor='#56B4E9')
        self._print(loop.body)

    def _print_Block(self, block):
        for node in block.args:
            self._print(node)
        parent = block.parent
        for node in block.args:
            self.dot.edge(self._nodeToStrFunction(parent), self._nodeToStrFunction(node))
            #parent = node

    def _print_SympyAssignment(self, assignment):
        self.dot.node(self._nodeToStrFunction(assignment))
        if self.full:
            for node in assignment.args:
                self._print(node)
            for node in assignment.args:
                self.dot.edge(self._nodeToStrFunction(assignment), self._nodeToStrFunction(node))

    def emptyPrinter(self, expr):
        if self.full:
            self.dot.node(self._nodeToStrFunction(expr))
            for node in expr.args:
                self._print(node)
            for node in expr.args:
                self.dot.edge(self._nodeToStrFunction(expr), self._nodeToStrFunction(node))
        else:
            raise NotImplemented('Dotprinter cannot print', expr)

    def doprint(self, expr):
        self._print(expr)
        return self.dot.source


def __shortened(node):
    from pystencils.astnodes import LoopOverCoordinate, KernelFunction, SympyAssignment
    if isinstance(node, LoopOverCoordinate):
        return "Loop over dim %d" % (node.coordinateToLoopOver,)
    elif isinstance(node, KernelFunction):
        params = [f.name for f in node.fieldsAccessed]
        params += [p.name for p in node.parameters if not p.isFieldArgument]
        return "Func: %s (%s)" % (node.functionName, ",".join(params))
    elif isinstance(node, SympyAssignment):
        return "Assignment: " + repr(node.lhs)


def dotprint(node, view=False, short=False, full=False, **kwargs):
    """
    Returns a string which can be used to generate a DOT-graph
    :param node: The ast which should be generated
    :param view: Boolen, if rendering of the image directly should occur.
    :param short: Uses the __shortened output
    :param full: Prints the whole tree with type information
    :param kwargs: is directly passed to the DotPrinter class: http://graphviz.readthedocs.io/en/latest/api.html#digraph
    :return: string in DOT format
    """
    nodeToStrFunction = __shortened if short else lambda expr: repr(type(expr)) + repr(expr) if full else repr
    printer = DotPrinter(nodeToStrFunction, full, **kwargs)
    dot = printer.doprint(node)
    if view:
        printer.dot.render(view=view)
    return dot

if __name__ == "__main__":
    from pystencils import Field
    import sympy as sp
    imgField = Field.createGeneric('I',
                                   spatialDimensions=2, # 2D image
                                   indexDimensions=1)   # multiple values per pixel: e.g. RGB
    w1, w2 = sp.symbols("w_1 w_2")
    sobelX = -w2 * imgField[-1, 0](1) - w1 * imgField[-1, -1](1) - w1 * imgField[-1, +1](1) \
             + w2 * imgField[+1, 0](1) + w1 * imgField[+1, -1](1) - w1 * imgField[+1, +1](1)
    sobelX

    dstField = Field.createGeneric('dst', spatialDimensions=2, indexDimensions=0)
    updateRule = sp.Eq(dstField[0, 0], sobelX)
    updateRule

    from pystencils.cpu import createKernel
    ast = createKernel([updateRule])
    print(dotprint(ast, short=True))
