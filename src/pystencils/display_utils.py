from typing import Any, Dict, Optional

import sympy as sp

from pystencils.backend import KernelFunction
from pystencils.kernel_wrapper import KernelWrapper


def to_dot(expr: sp.Expr, graph_style: Optional[Dict[str, Any]] = None, short=True):
    """Show a sympy or pystencils AST as dot graph"""
    from pystencils.sympyextensions.astnodes import Node
    try:
        import graphviz
    except ImportError:
        print("graphviz is not installed. Visualizing the AST is not available")
        return

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
    css_tag = f"<style>{css}</style>"
    display(HTML(css_tag))
    return HTML(highlight(code, CppLexer(), HtmlFormatter()))


def get_code_obj(ast: KernelWrapper | KernelFunction, custom_backend=None):
    """Returns an object to display generated code (C/C++ or CUDA)

    Can either be displayed as HTML in Jupyter notebooks or printed as normal string.
    """
    from pystencils.backend.emission import emit_code

    if isinstance(ast, KernelWrapper):
        ast = ast.ast

    class CodeDisplay:
        def __init__(self, ast_input):
            self.ast = ast_input

        def _repr_html_(self):
            return highlight_cpp(emit_code(self.ast)).__html__()

        def __str__(self):
            return emit_code(self.ast)

        def __repr__(self):
            return emit_code(self.ast)

    return CodeDisplay(ast)


def get_code_str(ast, custom_backend=None):
    return str(get_code_obj(ast, custom_backend))


def _isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def show_code(ast: KernelWrapper | KernelFunction, custom_backend=None):
    code = get_code_obj(ast, custom_backend)

    if _isnotebook():
        from IPython.display import display
        display(code)
    else:
        try:
            import rich.syntax
            import rich.console
            syntax = rich.syntax.Syntax(str(code), "c++", theme="monokai", line_numbers=True)
            console = rich.console.Console()
            console.print(syntax)
        except ImportError:
            print(code)
