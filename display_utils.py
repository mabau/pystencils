import sympy as sp
from typing import Any, Dict, Optional
from pystencils.astnodes import KernelFunction


def to_dot(expr: sp.Expr, graph_style: Optional[Dict[str, Any]] = None):
    """Show a sympy or pystencils AST as dot graph"""
    from pystencils.astnodes import Node
    import graphviz
    graph_style = {} if graph_style is None else graph_style

    if isinstance(expr, Node):
        from pystencils.backends.dot import print_dot
        return graphviz.Source(print_dot(expr, short=True, graph_attr=graph_style))
    else:
        from sympy.printing.dot import dotprint
        return graphviz.Source(dotprint(expr, graph_attr=graph_style))


def highlight_cpp(code: str):
    """Highlight the given C/C++ source code with pygments."""
    from IPython.display import HTML, display
    from pygments import highlight
    # noinspection PyUnresolvedReferences
    from pygments.formatters import HtmlFormatter
    # noinspection PyUnresolvedReferences
    from pygments.lexers import CppLexer

    css = HtmlFormatter().get_style_defs('.highlight')
    css_tag = f"<style>{css}</style>"
    display(HTML(css_tag))
    return HTML(highlight(code, CppLexer(), HtmlFormatter()))


def show_code(ast: KernelFunction):
    """Returns an object to display C code.

    Can either  be displayed as HTML in Jupyter notebooks or printed as normal string.
    """
    from pystencils.cpu import print_c

    class CodeDisplay:
        def __init__(self, ast_input):
            self.ast = ast_input

        def _repr_html_(self):
            return highlight_cpp(print_c(self.ast)).__html__()

        def __str__(self):
            return print_c(self.ast)

        def __repr__(self):
            return print_c(self.ast)
    return CodeDisplay(ast)
