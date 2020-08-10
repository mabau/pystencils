import sympy as sp

from pystencils.simp import subexpression_substitution_in_main_assignments
from pystencils.simp import add_subexpressions_for_divisions
from pystencils.simp import add_subexpressions_for_sums
from pystencils.simp import add_subexpressions_for_field_reads
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
