from sys import version_info as vs
import pytest

import pystencils.config
import sympy as sp
import pystencils as ps
import numpy as np

from pystencils.simp import subexpression_substitution_in_main_assignments
from pystencils.simp import add_subexpressions_for_divisions
from pystencils.simp import add_subexpressions_for_sums
from pystencils.simp import add_subexpressions_for_field_reads
from pystencils.simp.simplifications import add_subexpressions_for_constants
from pystencils import Assignment, AssignmentCollection, fields

a, b, c, d, x, y, z = sp.symbols("a b c d x y z")
s0, s1, s2, s3 = sp.symbols("s_:4")
f = sp.symbols("f_:9")


def test_subexpression_substitution_in_main_assignments():
    subexpressions = [
        Assignment(s0, 2 * a + 2 * b),
        Assignment(s1, 2 * a + 2 * b + 2 * c),
        Assignment(s2, 2 * a + 2 * b + 2 * c + 2 * d),
        Assignment(s3, 2 * a + 2 * b * c),
        Assignment(x, s1 + s2 + s0 + s3)
    ]
    main = [
        Assignment(f[0], s1 + s2 + s0 + s3),
        Assignment(f[1], s1 + s2 + s0 + s3),
        Assignment(f[2], s1 + s2 + s0 + s3),
        Assignment(f[3], s1 + s2 + s0 + s3),
        Assignment(f[4], s1 + s2 + s0 + s3)
    ]
    ac = AssignmentCollection(main, subexpressions)
    ac = subexpression_substitution_in_main_assignments(ac)
    for i in range(0, len(ac.main_assignments)):
        assert ac.main_assignments[i].rhs == x


def test_add_subexpressions_for_divisions():
    subexpressions = [
        Assignment(s0, 2 / a + 2 / b),
        Assignment(s1, 2 / a + 2 / b + 2 / c),
        Assignment(s2, 2 / a + 2 / b + 2 / c + 2 / d),
        Assignment(s3, 2 / a + 2 / b / c),
        Assignment(x, s1 + s2 + s0 + s3)
    ]
    main = [
        Assignment(f[0], s1 + s2 + s0 + s3)
    ]
    ac = AssignmentCollection(main, subexpressions)
    divs_before_optimisation = ac.operation_count["divs"]
    ac = add_subexpressions_for_divisions(ac)
    divs_after_optimisation = ac.operation_count["divs"]
    assert divs_before_optimisation - divs_after_optimisation == 8
    rhs = []
    for i in range(len(ac.subexpressions)):
        rhs.append(ac.subexpressions[i].rhs)

    assert 1/a in rhs
    assert 1/b in rhs
    assert 1/c in rhs
    assert 1/d in rhs


def test_add_subexpressions_for_constants():
    half = sp.Rational(1,2)
    sqrt_2 = sp.sqrt(2)
    main = [
        Assignment(f[0], half * a + half * b + half * c),
        Assignment(f[1], - half * a - half * b),
        Assignment(f[2], a * sqrt_2 - b * sqrt_2),
        Assignment(f[3], a**2 + b**2)
    ]
    ac = AssignmentCollection(main)
    ac = add_subexpressions_for_constants(ac)
    
    assert len(ac.subexpressions) == 2
    
    half_subexp = None
    sqrt_subexp = None

    for asm in ac.subexpressions:
        if asm.rhs == half:
            half_subexp = asm.lhs
        elif asm.rhs == sqrt_2:
            sqrt_subexp = asm.lhs
        else:
            pytest.fail(f"An unexpected subexpression was encountered: {asm}")
            
    assert half_subexp is not None
    assert sqrt_subexp is not None
    
    for asm in ac.main_assignments[:3]:
        assert isinstance(asm.rhs, sp.Mul)

    assert any(arg == half_subexp for arg in ac.main_assignments[0].rhs.args)
    assert any(arg == half_subexp for arg in ac.main_assignments[1].rhs.args)
    assert any(arg == sqrt_subexp for arg in ac.main_assignments[2].rhs.args)

    #   Do not replace exponents!
    assert ac.main_assignments[3].rhs == a**2 + b**2


def test_add_subexpressions_for_sums():
    subexpressions = [
        Assignment(s0, a + b + c + d),
        Assignment(s1, 3 * a * sp.sqrt(x) + 4 * b + c),
        Assignment(s2, 3 * a * sp.sqrt(x) + 4 * b + c),
        Assignment(s3, 3 * a * sp.sqrt(x) + 4 * b + c)
    ]
    main = [
        Assignment(f[0], s1 + s2 + s0 + s3)
    ]
    ac = AssignmentCollection(main, subexpressions)
    ops_before_optimisation = ac.operation_count
    ac = add_subexpressions_for_sums(ac)
    ops_after_optimisation = ac.operation_count
    assert ops_after_optimisation["adds"] == ops_before_optimisation["adds"]
    assert ops_after_optimisation["muls"] < ops_before_optimisation["muls"]
    assert ops_after_optimisation["sqrts"] < ops_before_optimisation["sqrts"]

    rhs = []
    for i in range(len(ac.subexpressions)):
        rhs.append(ac.subexpressions[i].rhs)

    assert a + b + c + d in rhs
    assert 3 * a * sp.sqrt(x) in rhs


def test_add_subexpressions_for_field_reads():
    s, v = fields("s(5), v(5): double[2D]")
    subexpressions = []
    main = [
        Assignment(s[0, 0](0), 3 * v[0, 0](0)),
        Assignment(s[0, 0](1), 10 * v[0, 0](1))
    ]
    ac = AssignmentCollection(main, subexpressions)
    assert len(ac.subexpressions) == 0
    ac = add_subexpressions_for_field_reads(ac)
    assert len(ac.subexpressions) == 2


@pytest.mark.parametrize('target', (ps.Target.CPU, ps.Target.GPU))
@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.skipif((vs.major, vs.minor, vs.micro) == (3, 8, 2), reason="does not work on python 3.8.2 for some reason")
def test_sympy_optimizations(target, dtype):
    if target == ps.Target.GPU:
        pytest.importorskip("pycuda")
    src, dst = ps.fields(f'src, dst:  {dtype}[2d]')

    assignments = ps.AssignmentCollection({
        src[0, 0]: 1.0 * (sp.exp(dst[0, 0]) - 1)
    })

    config = pystencils.config.CreateKernelConfig(target=target, default_number_float=dtype)
    ast = ps.create_kernel(assignments, config=config)

    ps.show_code(ast)

    code = ps.get_code_str(ast)
    if dtype == 'float32':
        assert 'expf(' in code
    elif dtype == 'float64':
        assert 'exp(' in code


@pytest.mark.parametrize('target', (ps.Target.CPU, ps.Target.GPU))
@pytest.mark.parametrize('simplification', (True, False))
@pytest.mark.skipif((vs.major, vs.minor, vs.micro) == (3, 8, 2), reason="does not work on python 3.8.2 for some reason")
def test_evaluate_constant_terms(target, simplification):
    if target == ps.Target.GPU:
        pytest.importorskip("pycuda")
    src, dst = ps.fields('src, dst:  float32[2d]')

    # cos of a number will always be simplified
    assignments = ps.AssignmentCollection({
        src[0, 0]: -sp.cos(1) + dst[0, 0]
    })

    config = pystencils.config.CreateKernelConfig(target=target, default_assignment_simplifications=simplification)
    ast = ps.create_kernel(assignments, config=config)
    code = ps.get_code_str(ast)
    assert 'cos(' not in code
