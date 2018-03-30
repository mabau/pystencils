
def toDot(expr, graphStyle={}):
    """Show a sympy or pystencils AST as dot graph"""
    from pystencils.astnodes import Node
    import graphviz
    if isinstance(expr, Node):
        from pystencils.backends.dot import dotprint
        return graphviz.Source(dotprint(expr, short=True, graph_attr=graphStyle))
    else:
        from sympy.printing.dot import dotprint
        return graphviz.Source(dotprint(expr, graph_attr=graphStyle))


def highlightCpp(code):
    """Highlight the given C/C++ source code with Pygments"""
    from IPython.display import HTML, display
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import CppLexer

    display(HTML("""
            <style>
            {pygments_css}
            </style>
            """.format(pygments_css=HtmlFormatter().get_style_defs('.highlight'))))
    return HTML(highlight(code, CppLexer(), HtmlFormatter()))


def showCode(ast):
    from pystencils.cpu import generateC

    class CodeDisplay:
        def __init__(self, astInput):
            self.ast = astInput

        def _repr_html_(self):
            return highlightCpp(generateC(self.ast)).__html__()

        def __str__(self):
            return generateC(self.ast)

        def __repr__(self):
            return generateC(self.ast)
    return CodeDisplay(ast)
