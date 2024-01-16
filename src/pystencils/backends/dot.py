import graphviz
try:
    from graphviz import Digraph
    import graphviz.quoting as quote
except ImportError:
    from graphviz import Digraph
    import graphviz.lang as quote
from sympy.printing.printer import Printer


# noinspection PyPep8Naming
class DotPrinter(Printer):
    """
    A printer which converts ast to DOT (graph description language).
    """
    def __init__(self, node_to_str_function, **kwargs):
        super(DotPrinter, self).__init__()
        self._node_to_str_function = node_to_str_function
        self.dot = Digraph(**kwargs)
        self.dot.quote_edge = quote.quote

    def _print_KernelFunction(self, func):
        self.dot.node(str(id(func)), style='filled', fillcolor='#a056db', label=self._node_to_str_function(func))
        self._print(func.body)
        self.dot.edge(str(id(func)), str(id(func.body)))

    def _print_LoopOverCoordinate(self, loop):
        self.dot.node(str(id(loop)), style='filled', fillcolor='#3498db', label=self._node_to_str_function(loop))
        self._print(loop.body)
        self.dot.edge(str(id(loop)), str(id(loop.body)))

    def _print_Block(self, block):
        for node in block.args:
            self._print(node)

        self.dot.node(str(id(block)), style='filled', fillcolor='#dbc256', label=repr(block))
        for node in block.args:
            self.dot.edge(str(id(block)), str(id(node)))

    def _print_SympyAssignment(self, assignment):
        self.dot.node(str(id(assignment)), style='filled', fillcolor='#56db7f',
                      label=self._node_to_str_function(assignment))

    def _print_Conditional(self, expr):
        self.dot.node(str(id(expr)), style='filled', fillcolor='#56bd7f', label=self._node_to_str_function(expr))
        self._print(expr.true_block)
        self.dot.edge(str(id(expr)), str(id(expr.true_block)))
        if expr.false_block:
            self._print(expr.false_block)
            self.dot.edge(str(id(expr)), str(id(expr.false_block)))

    def doprint(self, expr):
        self._print(expr)
        return self.dot.source


def __shortened(node):
    from pystencils.astnodes import LoopOverCoordinate, KernelFunction, SympyAssignment, Conditional
    if isinstance(node, LoopOverCoordinate):
        return "Loop over dim %d" % (node.coordinate_to_loop_over,)
    elif isinstance(node, KernelFunction):
        params = node.get_parameters()
        param_names = [p.field_name for p in params if p.is_field_pointer]
        param_names += [p.symbol.name for p in params if not p.is_field_parameter]
        return f"Func: {node.function_name} ({','.join(param_names)})"
    elif isinstance(node, SympyAssignment):
        return repr(node.lhs)
    elif isinstance(node, Conditional):
        return repr(node)
    else:
        raise NotImplementedError(f"Cannot handle node type {type(node)}")


def print_dot(node, view=False, short=False, **kwargs):
    """
    Returns a string which can be used to generate a DOT-graph
    :param node: The ast which should be generated
    :param view: Boolean, if rendering of the image directly should occur.
    :param short: Uses the __shortened output
    :param kwargs: is directly passed to the DotPrinter class: http://graphviz.readthedocs.io/en/latest/api.html#digraph
    :return: string in DOT format
    """
    node_to_str_function = repr
    if short:
        node_to_str_function = __shortened
    printer = DotPrinter(node_to_str_function, **kwargs)
    dot = printer.doprint(node)
    if view:
        return graphviz.Source(dot)
    return dot
