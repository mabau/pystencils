"""
Default Sympy optimizations applied in pystencils kernels using :func:`sympy.codegen.rewriting.optimize`.

See :func:`sympy.codegen.rewriting.optimize`.
"""


import itertools

from pystencils import Assignment
from pystencils.astnodes import SympyAssignment

try:
    from sympy.codegen.rewriting import optims_c99, optimize
    from sympy.codegen.rewriting import ReplaceOptim
    HAS_REWRITING = True

    # Evaluates all constant terms
    evaluate_constant_terms = ReplaceOptim(
        lambda e: hasattr(e, 'is_constant') and e.is_constant and not e.is_integer,
        lambda p: p.evalf()
    )

    optims_pystencils_cpu = [evaluate_constant_terms] + list(optims_c99)
    optims_pystencils_gpu = [evaluate_constant_terms] + list(optims_c99)
except ImportError:
    from warnings import warn
    warn("Could not import ReplaceOptim, optims_c99, optimize from sympy.codegen.rewriting."
         "Please update your sympy installation!")
    optims_c99 = []
    optims_pystencils_cpu = []
    optims_pystencils_gpu = []
    HAS_REWRITING = False


def optimize_assignments(assignments, optimizations):

    if HAS_REWRITING:
        assignments = [Assignment(a.lhs, optimize(a.rhs, optimizations))
                       if hasattr(a, 'lhs')
                       else a for a in assignments]
        assignments_nodes = [a.atoms(SympyAssignment) for a in assignments]
        for a in itertools.chain.from_iterable(assignments_nodes):
            a.optimize(optimizations)

    return assignments


def optimize_ast(ast, optimizations):

    if HAS_REWRITING:
        assignments_nodes = ast.atoms(SympyAssignment)
        for a in assignments_nodes:
            a.optimize(optimizations)

    return ast
