from typing import Any, Dict, Optional

import sympy as sp

from pystencils.astnodes import KernelFunction


def to_dot(expr: sp.Expr, graph_style: Optional[Dict[str, Any]] = None, short=True):
    """Show a sympy or pystencils AST as dot graph"""
    from pystencils.astnodes import Node
    import graphviz
    graph_style = {} if graph_style is None else graph_style

    if isinstance(expr, Node):
        from pystencils.backends.dot import print_dot
        return graphviz.Source(print_dot(expr, short=short, graph_attr=graph_style))
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
    css_tag = "<style>{css}</style>".format(css=css)
    display(HTML(css_tag))
    return HTML(highlight(code, CppLexer(), HtmlFormatter()))


def show_code(ast: KernelFunction, custom_backend=None):
    """Returns an object to display generated code (C/C++ or CUDA)

    Can either  be displayed as HTML in Jupyter notebooks or printed as normal string.
    """
    from pystencils.backends.cbackend import generate_c
    dialect = 'cuda' if ast.backend == 'gpucuda' else 'c'

    class CodeDisplay:
        def __init__(self, ast_input):
            self.ast = ast_input

        def _repr_html_(self):
            return highlight_cpp(generate_c(self.ast, dialect=dialect, custom_backend=custom_backend)).__html__()

        def __str__(self):
            return generate_c(self.ast, dialect=dialect, custom_backend=custom_backend)

        def __repr__(self):
            return generate_c(self.ast, dialect=dialect, custom_backend=custom_backend)
    return CodeDisplay(ast)
