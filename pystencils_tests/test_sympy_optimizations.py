import pytest
import sympy as sp

import pystencils
from pystencils.math_optimizations import HAS_REWRITING, optimize_assignments, optims_pystencils_cpu, optimize_ast


@pytest.mark.skipif(not HAS_REWRITING, reason="need sympy.codegen.rewriting")
def test_sympy_optimizations():
    for target in ('cpu', 'gpu'):
        for op_ast in (True, False):
            x, y, z = pystencils.fields('x, y, z:  float32[2d]')

            # Triggers Sympy's expm1 optimization
            # Sympy's expm1 optimization is tedious to use and the behaviour is highly depended on the sympy version. In
            # some cases the exp expression has to be encapsulated in brackets or multiplied with 1 or 1.0
            # for sympy to work properly ...
            assignments = pystencils.AssignmentCollection({
                x[0, 0]: 1.0 * (sp.exp(y[0, 0]) - 1)
            })

            if not op_ast:
                assignments = optimize_assignments(assignments, optims_pystencils_cpu)
            print(assignments)

            ast = pystencils.create_kernel(assignments, target=target)
            if op_ast:
                optimize_ast(ast, optims_pystencils_cpu)
            code = pystencils.get_code_str(ast)
            assert 'expm1(' in code


@pytest.mark.skipif(not HAS_REWRITING, reason="need sympy.codegen.rewriting")
def test_evaluate_constant_terms():
    for target in ('cpu', 'gpu'):
        x, y, z = pystencils.fields('x, y, z:  float32[2d]')

        # Triggers Sympy's cos optimization
        assignments = pystencils.AssignmentCollection({
            x[0, 0]: -sp.cos(1) + y[0, 0]
        })

        assignments = optimize_assignments(assignments, optims_pystencils_cpu)

        ast = pystencils.create_kernel(assignments, target=target)
        code = pystencils.get_code_str(ast)
        assert 'cos(' not in code
        print(code)


@pytest.mark.skipif(not HAS_REWRITING, reason="need sympy.codegen.rewriting")
def test_do_not_evaluate_constant_terms():
    optimizations = pystencils.math_optimizations.optims_pystencils_cpu
    optimizations.remove(pystencils.math_optimizations.evaluate_constant_terms)

    for target in ('cpu', 'gpu'):
        x, y, z = pystencils.fields('x, y, z:  float32[2d]')

        assignments = pystencils.AssignmentCollection({
            x[0, 0]: -sp.cos(1) + y[0, 0]
        })

        ast = pystencils.create_kernel(assignments, target=target)
        code = pystencils.get_code_str(ast)
        assert 'cos(' in code
        print(code)
