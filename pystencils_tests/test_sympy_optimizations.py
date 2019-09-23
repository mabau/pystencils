import pytest
import sympy as sp

import pystencils
from pystencils.math_optimizations import HAS_REWRITING, optimize_assignments, optims_pystencils_cpu


@pytest.mark.skipif(not HAS_REWRITING, reason="need sympy.codegen.rewriting")
def test_sympy_optimizations():
    for target in ('cpu', 'gpu'):
        x, y, z = pystencils.fields('x, y, z:  float32[2d]')

        # Triggers Sympy's expm1 optimization
        assignments = pystencils.AssignmentCollection({
            x[0, 0]: sp.exp(y[0, 0]) - 1
        })

        assignments = optimize_assignments(assignments, optims_pystencils_cpu)

        ast = pystencils.create_kernel(assignments, target=target)
        code = str(pystencils.show_code(ast))
        assert 'expm1(' in code


@pytest.mark.skipif(not HAS_REWRITING, reason="need sympy.codegen.rewriting")
def test_evaluate_constant_terms():
    for target in ('cpu', 'gpu'):
        x, y, z = pystencils.fields('x, y, z:  float32[2d]')

        # Triggers Sympy's expm1 optimization
        assignments = pystencils.AssignmentCollection({
            x[0, 0]: -sp.cos(1) + y[0, 0]
        })

        assignments = optimize_assignments(assignments, optims_pystencils_cpu)

        ast = pystencils.create_kernel(assignments, target=target)
        code = str(pystencils.show_code(ast))
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

        optimize_assignments(assignments, optimizations)

        ast = pystencils.create_kernel(assignments, target=target)
        code = str(pystencils.show_code(ast))
        assert 'cos(' in code
        print(code)
