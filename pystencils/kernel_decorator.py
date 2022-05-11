import ast
import inspect
import textwrap
from typing import Callable, Union, List, Dict, Tuple

import sympy as sp

from pystencils.assignment import Assignment
from pystencils.sympyextensions import SymbolCreator
from pystencils.config import CreateKernelConfig

__all__ = ['kernel', 'kernel_config']


def _kernel(func: Callable[..., None], **kwargs) -> Tuple[List[Assignment], str]:
    """
    Convenient function for kernel decorator to prevent code duplication
    Args:
        func: decorated function
        **kwargs: kwargs for the function
    Returns:
        assignments, function_name
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
    return assignments, func.__name__


def kernel(func: Callable[..., None], **kwargs) -> List[Assignment]:
    """Decorator to simplify generation of pystencils Assignments.

    Changes the meaning of the '@=' operator. Each line containing this operator gives a symbolic assignment
    in the result list. Furthermore the meaning of the ternary inline 'if-else' changes meaning to denote a
    sympy Piecewise.

    The decorated function may not receive any arguments, with exception of an argument called 's' that specifies
    a SymbolCreator()
    Args:
        func: decorated function
        **kwargs: kwargs for the function

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
    assignments, _ = _kernel(func, **kwargs)
    return assignments


def kernel_config(config: CreateKernelConfig, **kwargs) -> Callable[..., Dict]:
    """Decorator to simplify generation of pystencils Assignments, which takes a configuration
    and updates the function name accordingly.

    Changes the meaning of the '@=' operator. Each line containing this operator gives a symbolic assignment
    in the result list. Furthermore, the meaning of the ternary inline 'if-else' changes meaning to denote a
    sympy Piecewise.

    The decorated function may not receive any arguments, with exception to an argument called 's' that specifies
    a SymbolCreator()
    Args:
        config: Specify whether to return the list with assignments, or a dictionary containing additional settings
                like func_name
    Returns:
        decorator with config

    Examples:
        >>> import pystencils as ps
        >>> kernel_configuration = ps.CreateKernelConfig()
        >>> @kernel_config(kernel_configuration)
        ... def my_kernel(s):
        ...     src, dst = ps.fields('src, dst: [2D]')
        ...     s.neighbors @= src[0, 1] + src[1, 0]
        ...     dst[0, 0]      @= s.neighbors + src[0, 0] if src[0, 0] > 0 else 0
        >>> f, g = ps.fields('src, dst: [2D]')
        >>> assert my_kernel['assignments'][0].rhs == f[0, 1] + f[1, 0]
    """
    def decorator(func: Callable[..., None]) -> Union[List[Assignment], Dict]:
        """
        Args:
            func: decorated function
        Returns:
            Dict for unpacking into create_kernel
        """
        assignments, func_name = _kernel(func, **kwargs)
        config.function_name = func_name
        return {'assignments': assignments, 'config': config}
    return decorator


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
