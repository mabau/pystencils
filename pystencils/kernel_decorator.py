import ast
import inspect
import textwrap

import sympy as sp

from pystencils.assignment import Assignment
from pystencils.sympyextensions import SymbolCreator

__all__ = ['kernel']


def kernel(func, **kwargs):
    """Decorator to simplify generation of pystencils Assignments.

    Changes the meaning of the '@=' operator. Each line containing this operator gives a symbolic assignment
    in the result list. Furthermore the meaning of the ternary inline 'if-else' changes meaning to denote a
    sympy Piecewise.

    The decorated function may not receive any arguments, with exception of an argument called 's' that specifies
    a SymbolCreator()

    Examples:
        >>> import pystencils as ps
        >>> @kernel
        ... def my_kernel(s):
        ...     f, g = ps.fields('f, g: [2D]')
        ...     s.neighbors @= f[0,1] + f[1,0]
        ...     g[0,0]      @= s.neighbors + f[0,0] if f[0,0] > 0 else 0
        >>> f, g = ps.fields('f, g: [2D]')
        >>> assert my_kernel[0].rhs == f[0,1] + f[1,0]
    """
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    a = ast.parse(source)
    KernelFunctionRewrite().visit(a)
    ast.fix_missing_locations(a)
    gl = func.__globals__.copy()

    assignments = []

    def assignment_adder(lhs, rhs):
        assignments.append(Assignment(lhs, rhs))

    gl['_add_assignment'] = assignment_adder
    gl['_Piecewise'] = sp.Piecewise
    gl.update(inspect.getclosurevars(func).nonlocals)
    exec(compile(a, filename="<ast>", mode="exec"), gl)
    func = gl[func.__name__]
    args = inspect.getfullargspec(func).args
    if 's' in args and 's' not in kwargs:
        kwargs['s'] = SymbolCreator()
    func(**kwargs)
    return assignments


# noinspection PyMethodMayBeStatic
class KernelFunctionRewrite(ast.NodeTransformer):

    def visit_IfExp(self, node):
        piecewise_func = ast.Name(id='_Piecewise', ctx=ast.Load())
        piecewise_func = ast.copy_location(piecewise_func, node)
        piecewise_args = [ast.Tuple(elts=[node.body, node.test], ctx=ast.Load()),
                          ast.Tuple(elts=[node.orelse, ast.NameConstant(value=True)], ctx=ast.Load())]
        result = ast.Call(func=piecewise_func, args=piecewise_args, keywords=[])

        return ast.copy_location(result, node)

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        node.target.ctx = ast.Load()
        new_node = ast.Expr(ast.Call(func=ast.Name(id='_add_assignment', ctx=ast.Load()),
                                     args=[node.target, node.value],
                                     keywords=[]))
        return ast.copy_location(new_node, node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.decorator_list = []
        return node
