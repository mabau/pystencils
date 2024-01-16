from typing import List, Union

import sympy as sp

from pystencils.astnodes import Node
from pystencils.simp import AssignmentCollection
from pystencils.assignment import Assignment


# noinspection PyPep8Naming
class fast_division(sp.Function):
    """
    Produces special float instructions for CUDA kernels
    """
    nargs = (2,)


# noinspection PyPep8Naming
class fast_sqrt(sp.Function):
    """
    Produces special float instructions for CUDA kernels
    """
    nargs = (1, )


# noinspection PyPep8Naming
class fast_inv_sqrt(sp.Function):
    """
    Produces special float instructions for CUDA kernels
    """
    nargs = (1, )


def _run(term, visitor):
    if isinstance(term, AssignmentCollection):
        new_main_assignments = _run(term.main_assignments, visitor)
        new_subexpressions = _run(term.subexpressions, visitor)
        return term.copy(new_main_assignments, new_subexpressions)
    elif isinstance(term, list):
        return [_run(e, visitor) for e in term]
    else:
        return visitor(term)


def insert_fast_sqrts(term: Union[sp.Expr, List[sp.Expr], AssignmentCollection, Assignment]):
    def visit(expr):
        if isinstance(expr, Node):
            return expr
        if expr.func == sp.Pow and isinstance(expr.exp, sp.Rational) and expr.exp.q == 2:
            power = expr.exp.p
            if power < 0:
                return fast_inv_sqrt(expr.args[0]) ** (-power)
            else:
                return fast_sqrt(expr.args[0]) ** power
        else:
            new_args = [visit(a) for a in expr.args]
            return expr.func(*new_args) if new_args else expr
    return _run(term, visit)


def insert_fast_divisions(term: Union[sp.Expr, List[sp.Expr], AssignmentCollection, Assignment]):

    def visit(expr):
        if isinstance(expr, Node):
            return expr
        if expr.func == sp.Mul:
            div_args = []
            other_args = []
            for a in expr.args:
                if a.func == sp.Pow and a.exp.is_integer and a.exp < 0:
                    div_args.append(visit(a.base) ** (-a.exp))
                else:
                    other_args.append(visit(a))
            if div_args:
                return fast_division(sp.Mul(*other_args), sp.Mul(*div_args))
            else:
                return sp.Mul(*other_args)
        elif expr.func == sp.Pow and expr.exp.is_integer and expr.exp < 0:
            return fast_division(1, visit(expr.base) ** (-expr.exp))
        else:
            new_args = [visit(a) for a in expr.args]
            return expr.func(*new_args) if new_args else expr

    return _run(term, visit)
